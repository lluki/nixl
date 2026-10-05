/* SPDX-License-Identifier: Apache-2.0 */
#include "nixlshard/g3.h"
#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <deque>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <limits>
#include <linux/fs.h>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>

namespace nixlshard {
namespace {
using Clock = std::chrono::steady_clock;
uint64_t ns(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count();
}
struct Finally { std::function<void()> fn; ~Finally() { if (fn) fn(); } };
struct Aligned {
    void *pointer = nullptr;
    size_t bytes;
    Aligned(size_t alignment, size_t length) : bytes(length) {
        if (::posix_memalign(&pointer, alignment, length)) throw std::bad_alloc();
    }
    ~Aligned() { std::free(pointer); }
};
size_t total(const std::vector<G3Buffer> &buffers) {
    if (buffers.empty()) throw std::invalid_argument("empty G3 buffer list");
    size_t result=0;
    for (const auto &b:buffers) {
        if (!b.address || !b.bytes || b.address>UINTPTR_MAX-b.bytes ||
            result>SIZE_MAX-b.bytes || (b.type!=DRAM_SEG && b.type!=VRAM_SEG))
            throw std::invalid_argument("invalid G3 memory descriptor");
        result+=b.bytes;
    }
    return result;
}
AllocationIdentity identity(const Allocation &a) { return {a.key,a.id,a.generation,a.record}; }
int disk_numa(const G3DeviceConfig &config) {
    if (config.numa_node>=0) return config.numa_node;
    struct stat st{};
    if (::stat(config.disk.path.c_str(),&st) || !S_ISBLK(st.st_mode))
        throw std::invalid_argument("debug file requires explicit NUMA node");
    auto link="/sys/dev/block/"+std::to_string(major(st.st_rdev))+":"+std::to_string(minor(st.st_rdev));
    auto path=std::filesystem::canonical(link);
    for (;path!=path.root_path();path=path.parent_path()) {
        std::ifstream input(path/"numa_node"); int node=-1;
        if (input>>node && node>=0) return node;
    }
    throw std::invalid_argument("assigned block device has unknown NUMA topology");
}
thread_local const void *metric_owner=nullptr;
thread_local G3Metrics *metric_target=nullptr;
thread_local G3Deadline metric_deadline=G3Deadline::max();
thread_local const void *admission_owner=nullptr, *admission_leases=nullptr;
struct MetricScope {
    const void *old_owner=metric_owner; G3Metrics *old_target=metric_target;
    G3Deadline old_deadline=metric_deadline;
    MetricScope(const void *owner,G3Metrics &target,G3Deadline until) {
        metric_owner=owner;metric_target=&target;metric_deadline=until;
    }
    ~MetricScope() { metric_owner=old_owner;metric_target=old_target;metric_deadline=old_deadline; }
};
}
struct G3TransferLayer::Impl {
    G3Config config;
    G3Context context;
    struct Device { G3DeviceConfig config; int numa=-1; std::unique_ptr<DiskIndex> index; std::string error; };
    std::vector<Device> devices;
    struct Registration {
        G3MemoryHandle id=0;
        std::vector<G3Buffer> buffers;
        nixl_reg_dlist_t descriptor;
        size_t references=0;
        bool owned=false,retiring=false;
        explicit Registration(nixl_mem_t type):descriptor(type) {}
    };
    mutable std::mutex mutex;
    std::mutex register_mutex,close_mutex;
    std::condition_variable idle;
    bool stopping=false,closed=false,close_recorded=false;
    size_t active=0;
    uint64_t next_registration=1;
    std::unordered_map<G3MemoryHandle,std::shared_ptr<Registration>> registrations;
    std::unordered_map<int,nixl_reg_dlist_t> files;
    std::set<std::string> pending;
    G3CloseResult close_result;

    struct Operation {
        Impl &owner; bool admitted=false;
        explicit Operation(Impl &i,bool bounded=true):owner(i) {
            std::lock_guard lock(i.mutex);
            if (!i.stopping && (!bounded || i.active<i.config.max_active)) { ++i.active;admitted=true; }
        }
        ~Operation() {
            if (admitted) { std::lock_guard lock(owner.mutex);--owner.active;owner.idle.notify_all(); }
        }
    };
    struct Leases {
        Impl &owner;std::vector<std::shared_ptr<Registration>> refs;
        explicit Leases(Impl &i):owner(i) {}
        ~Leases() {
            std::lock_guard lock(owner.mutex);
            for (auto &r:refs) --r->references;
            owner.idle.notify_all();
        }
    };
    struct Temporary {
        Impl &owner;
        nixl_reg_dlist_t descriptor{DRAM_SEG};
        explicit Temporary(Impl &i,void *address,size_t bytes):owner(i) {
            descriptor.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(address),bytes,0,""));
            std::lock_guard lock(i.register_mutex);
            if(i.context.agent->registerMem(descriptor,&i.context.file_options)!=NIXL_SUCCESS)
                throw std::runtime_error("G3 temporary DRAM registration failed");
        }
        ~Temporary() {
            std::lock_guard lock(owner.register_mutex);
            if(owner.context.agent->deregisterMem(descriptor,&owner.context.file_options)!=NIXL_SUCCESS)
                std::cerr<<"G3 temporary registration cleanup failed\n";
        }
    };
    G3Deadline deadline(G3Deadline value) const {
        return std::min(value,Clock::now()+std::chrono::milliseconds(config.timeout_ms));
    }
    Status transfer(bool write,const nixl_xfer_dlist_t &memory,const nixl_xfer_dlist_t &file,
                    G3Deadline until) {
        if(Clock::now()>=until) return Status::timeout;
        nixlXferReqH *request=nullptr;
        auto options=context.file_options;
        auto rc=context.agent->createXferReq(write?NIXL_WRITE:NIXL_READ,memory,file,
                                             context.agent_name,request,&options);
        if(rc!=NIXL_SUCCESS) return Status::io_error;
        rc=context.agent->postXferReq(request,&options);
        bool expired=false;
        // Logical timeout never releases claims or memory while I/O can still touch
        // them. The facade can expose its deadline separately while this worker drains.
        while(rc==NIXL_IN_PROG) {
            expired|=Clock::now()>=until;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            rc=context.agent->getXferStatus(request);
        }
        auto release=context.agent->releaseXferReq(request);
        if(release!=NIXL_SUCCESS)
            throw std::runtime_error("G3 backend failed terminal request release");
        if(expired || Clock::now()>=until) return Status::timeout;
        return rc==NIXL_SUCCESS?Status::success:Status::io_error;
    }
    void file_registration(int fd) {
        std::lock_guard lock(register_mutex);
        if(files.contains(fd)) return;
        struct stat st{};uint64_t capacity=0;
        if(::fstat(fd,&st)) throw std::runtime_error("G3 cannot inspect opened descriptor");
        if(S_ISBLK(st.st_mode)) {
            if(::ioctl(fd,BLKGETSIZE64,&capacity)) throw std::runtime_error("G3 block capacity query failed");
        } else capacity=st.st_size;
        if(!capacity) throw std::runtime_error("G3 cannot register empty storage");
        nixl_reg_dlist_t descriptor(FILE_SEG);
        descriptor.addDesc(nixlBlobDesc(0,capacity,fd,""));
        if(context.agent->registerMem(descriptor,&context.file_options)!=NIXL_SUCCESS)
            throw std::runtime_error("G3 storage registration failed");
        try { files.emplace(fd,descriptor); }
        catch(...) { context.agent->deregisterMem(descriptor,&context.file_options);throw; }
    }
    Status retire_file(int fd) {
        std::lock_guard lock(register_mutex);
        auto it=files.find(fd);
        if(it==files.end()) return Status::success;
        if(context.agent->deregisterMem(it->second,&context.file_options)!=NIXL_SUCCESS)
            return Status::io_error;
        files.erase(it);
        return Status::success;
    }
    void metadata_account(bool write,Clock::time_point start,size_t bytes,bool completed) {
        if(metric_owner!=this || !metric_target)return;
        const auto elapsed=ns(start);auto &m=*metric_target;m.metadata_ns+=elapsed;
        if(write)m.metadata_write_ns+=elapsed;else m.metadata_read_ns+=elapsed;
        if(completed) {
            m.metadata_bytes+=bytes;
            if(write)m.metadata_write_bytes+=bytes;else m.metadata_read_bytes+=bytes;
        }
    }
    Status metadata(int fd,bool write,uint64_t offset,void *buffer,size_t bytes) {
        const auto start=Clock::now();
        try {
            file_registration(fd);
            Temporary registration(*this,buffer,bytes);
            nixl_xfer_dlist_t memory(DRAM_SEG),file(FILE_SEG);
            memory.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(buffer),bytes,0));
            file.addDesc(nixlBasicDesc(offset,bytes,fd));
            auto result=transfer(write,memory,file,deadline(metric_owner==this?metric_deadline:G3Deadline::max()));
            metadata_account(write,start,bytes,result==Status::success);
            return result;
        } catch(...) {
            metadata_account(write,start,bytes,false);
            return Status::io_error;
        }
    }
    Status metadata_batch(int fd,const std::vector<MetadataRead> &ranges) {
        if(ranges.empty())return Status::success;
        const auto start=Clock::now();size_t bytes=0;
        try {
            if(ranges.size()>8)throw std::invalid_argument("too many G3 metadata ranges");
            auto base=reinterpret_cast<uintptr_t>(ranges.front().buffer);
            for(const auto &range:ranges) {
                if(reinterpret_cast<uintptr_t>(range.buffer)!=base+bytes || range.bytes>128*1024*1024-bytes)
                    throw std::invalid_argument("G3 metadata batch scratch is not bounded contiguous memory");
                bytes+=range.bytes;
            }
            file_registration(fd);
            Temporary registration(*this,reinterpret_cast<void*>(base),bytes);
            nixl_xfer_dlist_t memory(DRAM_SEG),file(FILE_SEG);
            for(const auto &range:ranges) {
                memory.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(range.buffer),range.bytes,0));
                file.addDesc(nixlBasicDesc(range.offset,range.bytes,fd));
            }
            auto result=transfer(false,memory,file,deadline(metric_owner==this?metric_deadline:G3Deadline::max()));
            metadata_account(false,start,bytes,result==Status::success);
            return result;
        } catch(...) { metadata_account(false,start,bytes,false);return Status::io_error; }
    }
    explicit Impl(G3Config cfg,G3Context ctx):config(std::move(cfg)),context(std::move(ctx)) {
        if(config.instance_id.empty() || config.instance_id.size()>512 || config.devices.empty() || config.devices.size()>128 ||
           !config.max_active || !config.staging_bytes || !config.timeout_ms ||
           !context.agent || context.agent_name.empty() || context.file_options.backends.empty())
            throw std::invalid_argument("invalid G3 instance/context configuration");
        if(context.memory_options.backends.empty()) context.memory_options=context.file_options;
        const auto &format=config.devices.front().disk;
        if(format.namespace_id.empty())throw std::invalid_argument("G3 requires an exact namespace before opening devices");
        for(const auto &device:config.devices) {
            const auto &disk=device.disk;
            if(disk.namespace_id!=format.namespace_id || disk.key_bytes!=format.key_bytes ||
               disk.min_object_bytes!=format.min_object_bytes || disk.max_object_bytes!=format.max_object_bytes)
                throw std::invalid_argument("G3 instance devices must share namespace and object geometry");
        }
        devices.reserve(config.devices.size());
        for(const auto &dc:config.devices) {
            Device d;d.config=dc;
            try {
                d.numa=disk_numa(dc);
                d.index=std::make_unique<DiskIndex>(dc.disk,
                    [this](int fd,bool write,uint64_t offset,void *buffer,size_t bytes) {
                        return metadata(fd,write,offset,buffer,bytes);
                    },
                    [](int fd) {
                        int rc;do { rc=::fdatasync(fd); }while(rc<0 && errno==EINTR);
                        return rc==0?Status::success:Status::io_error;
                    },config.events,nullptr,[this](int fd){return retire_file(fd);},
                    [this](int fd,const std::vector<MetadataRead>&ranges){return metadata_batch(fd,ranges);});
            } catch(const std::exception &e) {
                d.error=e.what();std::cerr<<"G3 excluded "<<dc.disk.path<<": "<<d.error<<"\n";
            }
            devices.push_back(std::move(d));
        }
        if(std::none_of(devices.begin(),devices.end(),[](const Device &d){return bool(d.index);}))
            throw std::runtime_error("G3 has no usable assigned devices");
    }
    ~Impl() {
        // Descriptor callbacks need the registry/mutex/context still alive.
        for(auto &disk:devices)disk.index.reset();
    }
    G3MemoryHandle add_registration(const std::vector<G3Buffer> &buffers,bool owned) {
        total(buffers);
        for(size_t a=0;a<buffers.size();++a)for(size_t b=a+1;b<buffers.size();++b)
            if(buffers[a].type==buffers[b].type && buffers[a].device==buffers[b].device &&
               buffers[a].address<buffers[b].address+buffers[b].bytes &&
               buffers[b].address<buffers[a].address+buffers[a].bytes)
                throw std::invalid_argument("overlapping descriptors in G3 registration");
        const auto type=buffers.front().type;
        for(const auto &b:buffers) if(b.type!=type)
            throw std::invalid_argument("one G3 registration cannot mix memory types");
        auto r=std::make_shared<Registration>(type);r->buffers=buffers;r->owned=owned;
        for(const auto &b:buffers)r->descriptor.addDesc(nixlBlobDesc(b.address,b.bytes,b.device,""));
        std::unique_lock lock(mutex);
        if(stopping) throw std::runtime_error("G3 instance is closing");
        for(const auto &[id,existing]:registrations) {
            (void)id;
            for(const auto &a:existing->buffers)for(const auto &b:buffers)
                if(a.type==b.type && a.device==b.device && a.address<b.address+b.bytes &&
                   b.address<a.address+a.bytes) throw std::invalid_argument("overlapping G3 registration");
        }
        if(next_registration==UINT64_MAX) throw std::runtime_error("G3 registration identifiers exhausted");
        if(owned) {
            std::lock_guard control(register_mutex);
            if(context.agent->registerMem(r->descriptor,&context.memory_options)!=NIXL_SUCCESS)
                throw std::runtime_error("G3 caller memory registration unsupported/failed");
        }
        r->id=next_registration++;
        try { registrations.emplace(r->id,r); }
        catch(...) {
            if(owned)context.agent->deregisterMem(r->descriptor,&context.memory_options);
            throw;
        }
        return r->id;
    }
    Status lease(const std::vector<G3Buffer> &buffers,Leases &leases,
                 std::vector<std::unique_ptr<Temporary>> &temporary) {
        std::unique_lock lock(mutex);
        for(const auto &b:buffers) {
            std::shared_ptr<Registration> found;
            for(const auto &[id,r]:registrations) {
                if(b.registration && b.registration!=id)continue;
                for(const auto &range:r->buffers) {
                    if(range.type==b.type && range.device==b.device &&
                       range.address<=b.address && b.bytes<=range.bytes &&
                       b.address-range.address<=range.bytes-b.bytes) {found=r;break;}
                }
                if(found)break;
            }
            if(found) {
                if(found->retiring) {
                    const auto *admitted=admission_owner==this?static_cast<const Leases *>(admission_leases):nullptr;
                    if(!admitted || std::find(admitted->refs.begin(),admitted->refs.end(),found)==admitted->refs.end())
                        return Status::not_ready;
                }
                leases.refs.push_back(found);++found->references;
            } else {
                if(b.registration || config.memory_mode==MemoryMode::explicit_registration)
                    return Status::invalid_input;
                // Automatic VRAM registration is valid only on a capable payload
                // backend; this initial POSIX path rejects it before dereferencing.
                if(b.type!=DRAM_SEG) return Status::invalid_input;
                lock.unlock();
                temporary.push_back(std::make_unique<Temporary>(*this,reinterpret_cast<void*>(b.address),b.bytes));
                lock.lock();
            }
        }
        return Status::success;
    }
    bool eligible(const Device &disk,const Allocation &a,const std::vector<G3Buffer> &buffers)const {
        const auto unit=disk.index->unit_bytes();
        if(a.slots.size()>SIZE_MAX/unit || a.slots.size()*unit!=a.bytes)return false;
        for(const auto &b:buffers) {
            if(b.type!=DRAM_SEG)return false; // POSIX advertises DRAM+FILE only.
            if(disk.config.disk.direct_io && (b.address%4096 || b.bytes%4096))return false;
        }
        return true;
    }
    void descriptors(const Device &disk,const Allocation &a,const std::vector<G3Buffer> &buffers,
                     nixl_xfer_dlist_t &memory,nixl_xfer_dlist_t &file) {
        size_t segment=0,within=0;
        const auto unit=disk.index->unit_bytes();
        for(size_t cursor=0;cursor<a.slots.size();) {
            auto end=cursor+1;
            while(end<a.slots.size() && a.slots[end]==a.slots[end-1]+1)++end;
            const auto bytes=(end-cursor)*unit;
            for(size_t offset=0;offset<bytes;) {
                const auto &b=buffers.at(segment);
                const auto length=std::min(bytes-offset,b.bytes-within);
                memory.addDesc(nixlBasicDesc(b.address+within,length,b.device));
                file.addDesc(nixlBasicDesc(disk.index->slot_offset(a.slots[cursor])+offset,length,disk.index->fd()));
                offset+=length;within+=length;
                if(within==b.bytes){++segment;within=0;}
            }
            cursor=end;
        }
    }
    void fail_payload(size_t id,Status result) {
        if(result==Status::io_error) devices.at(id).index->set_state(DeviceState::failed);
    }
};

G3TransferLayer::G3TransferLayer(G3Config config,G3Context context)
    :impl_(std::make_unique<Impl>(std::move(config),std::move(context))) {}
G3TransferLayer::~G3TransferLayer() { try {close(CloseMode::discard);}catch(...) {} }
G3MemoryHandle G3TransferLayer::register_memory(const std::vector<G3Buffer>&b) {
    return impl_->add_registration(b,true);
}
G3MemoryHandle G3TransferLayer::borrow_registered_memory(const std::vector<G3Buffer>&b) {
    return impl_->add_registration(b,false);
}
Status G3TransferLayer::deregister_memory(G3MemoryHandle id) {
    return finish_deregister(id,false);
}
Status G3TransferLayer::finish_deregister(G3MemoryHandle id,bool already_retiring) {
    auto &i=*impl_;std::unique_lock lock(i.mutex);auto it=i.registrations.find(id);
    if(it==i.registrations.end()) return Status::invalid_input;
    auto r=it->second;
    if(r->retiring&&!already_retiring)return Status::busy;
    r->retiring=true;i.idle.wait(lock,[&]{return r->references==0;});
    if(r->owned) {
        std::lock_guard control(i.register_mutex);
        if(i.context.agent->deregisterMem(r->descriptor,&i.context.memory_options)!=NIXL_SUCCESS)
            return Status::io_error; // retained, retired registration, never reused
    }
    i.registrations.erase(id);return Status::success;
}
std::shared_ptr<void> G3TransferLayer::acquire_async_use(const std::vector<G3Buffer> &buffers) {
    total(buffers);auto &i=*impl_;auto guard=std::make_shared<Impl::Leases>(i);
    std::lock_guard lock(i.mutex);
    if(i.stopping)throw std::runtime_error("G3 instance closing");
    for(const auto &b:buffers) {
        std::shared_ptr<Impl::Registration> found;
        for(const auto &[id,r]:i.registrations) {
            if(b.registration&&id!=b.registration)continue;
            for(const auto &range:r->buffers)if(range.type==b.type && range.device==b.device &&
                range.address<=b.address && b.bytes<=range.bytes &&
                b.address-range.address<=range.bytes-b.bytes){found=r;break;}
            if(found)break;
        }
        if(found) {
            if(found->retiring)throw std::runtime_error("G3 registration retiring");
            guard->refs.push_back(found);++found->references;
        } else if(b.registration || i.config.memory_mode==MemoryMode::explicit_registration)
            throw std::invalid_argument("G3 asynchronous operation requires declared memory");
    }
    return guard;
}
void G3TransferLayer::run_with_async_use(const std::shared_ptr<void> &guard,const std::function<void()> &fn) {
    const auto *old_owner=admission_owner,*old_leases=admission_leases;
    admission_owner=guard?impl_.get():nullptr;admission_leases=guard.get();
    Finally restore{[&]{admission_owner=old_owner;admission_leases=old_leases;}};fn();
}
Status G3TransferLayer::start_async_deregister(G3MemoryHandle id) {
    auto &i=*impl_;std::lock_guard lock(i.mutex);auto it=i.registrations.find(id);
    if(it==i.registrations.end())return Status::invalid_input;
    if(it->second->retiring)return Status::busy;
    it->second->retiring=true;return Status::success;
}
void G3TransferLayer::cancel_async_deregister(G3MemoryHandle id) {
    auto &i=*impl_;std::lock_guard lock(i.mutex);auto it=i.registrations.find(id);
    if(it!=i.registrations.end())it->second->retiring=false;
}
G3Result G3TransferLayer::write(const std::string &key,const std::vector<G3Buffer>&buffers,
                               int numa,G3Deadline until) {
    auto &i=*impl_;Impl::Operation op(i);G3Result result;
    if(!op.admitted){result.status=Status::busy;return result;}
    if(numa<0)return result;
    try {result.bytes=total(buffers);}catch(const std::invalid_argument&){return result;}
    for(const auto &b:buffers)if(b.type!=DRAM_SEG)return result;
    until=i.deadline(until);MetricScope metrics(&i,result.metrics,until);
    Impl::Leases leases(i);std::vector<std::unique_ptr<Impl::Temporary>> temporaries;
    result.status=i.lease(buffers,leases,temporaries);
    if(result.status!=Status::success)return result;
    {std::lock_guard lock(i.mutex);if(!i.pending.insert(key).second){result.status=Status::busy;return result;}}
    Finally pending{[&]{std::lock_guard lock(i.mutex);i.pending.erase(key);}};
    // Immutable identity is instance-wide, even if it was placed on a later disk.
    for(size_t d=0;d<i.devices.size();++d) {
        auto &disk=i.devices[d];
        if(!disk.index || !disk.index->exists(key))continue;
        Allocation existing;result.status=disk.index->pin(key,existing);
        if(result.status!=Status::success)return result;
        Finally claim{[&]{disk.index->unpin(existing);}};
        result.device_index=d;result.identity=identity(existing);
        result.status=existing.bytes==result.bytes?Status::success:Status::invalid_input;
        return result;
    }
    Allocation allocation;size_t selected=i.devices.size();bool busy=false;
    // Scan all assigned healthy local disks before reclamation/backoff.
    while(Clock::now()<until) {
        bool suitable=false,assigned=false,failed=false;
        for(size_t d=0;d<i.devices.size();++d) {
            auto &disk=i.devices[d];
            if(disk.numa!=numa)continue;
            assigned=true;
            if(!disk.index || disk.index->state()==DeviceState::failed){failed=true;continue;}
            if(disk.index->state()!=DeviceState::active)continue;
            suitable=true;
            auto status=disk.index->reserve(key,result.bytes,allocation);
            if(status==Status::success){selected=d;break;}
            if(status==Status::invalid_input){result.status=status;return result;}
            busy|=status==Status::busy;
        }
        if(selected!=i.devices.size())break;
        if(!suitable){result.status=!assigned?Status::invalid_input:failed?Status::io_error:Status::not_ready;return result;}
        bool reclaimed=false;
        for(auto &disk:i.devices)if(disk.index && disk.numa==numa &&
                                    disk.index->state()==DeviceState::active) {
            const auto status=disk.index->evict_one();
            reclaimed|=status==Status::success;busy|=status==Status::busy;
        }
        if(!reclaimed){result.status=busy?Status::busy:Status::no_space;return result;}
    }
    if(selected==i.devices.size()){result.status=Status::timeout;return result;}
    auto &disk=i.devices[selected];result.device_index=selected;result.identity=identity(allocation);
    if(allocation.already_present){result.status=Status::success;return result;}
    Finally reservation{[&]{disk.index->abort(allocation);}};
    std::unique_ptr<Aligned> staged;std::unique_ptr<Impl::Temporary> staged_registration;
    auto targets=buffers;result.direct=i.eligible(disk,allocation,buffers);
    if(!result.direct) {
        const auto bytes=allocation.slots.size()*disk.index->unit_bytes();
        if(bytes>i.config.staging_bytes){result.status=Status::no_space;return result;}
        staged=std::make_unique<Aligned>(4096,bytes);std::memset(staged->pointer,0,bytes);
        const auto copy_start=Clock::now();
        size_t pos=0;for(const auto &b:buffers){std::memcpy(static_cast<char*>(staged->pointer)+pos,
                                                           reinterpret_cast<void*>(b.address),b.bytes);pos+=b.bytes;}
        result.metrics.copy_ns=ns(copy_start);
        result.metrics.copy_bytes=result.bytes;
        staged_registration=std::make_unique<Impl::Temporary>(i,staged->pointer,bytes);
        targets={{reinterpret_cast<uintptr_t>(staged->pointer),bytes,DRAM_SEG,0,0}};
    }
    nixl_xfer_dlist_t memory(DRAM_SEG),file(FILE_SEG);
    i.descriptors(disk,allocation,targets,memory,file);
    const auto start=Clock::now();result.status=i.transfer(true,memory,file,until);
    result.metrics.payload_ns=ns(start);
    if(result.status==Status::success)
        result.metrics.payload_bytes=allocation.slots.size()*disk.index->unit_bytes();
    if(result.status==Status::success)result.status=disk.index->publish(allocation);
    i.fail_payload(selected,result.status);
    if(result.status==Status::success)reservation.fn={};
    return result;
}
G3BatchResult G3TransferLayer::read_batch(const std::vector<G3Read>&reads,int numa,G3Deadline until) {
    auto &i=*impl_;Impl::Operation op(i);G3BatchResult batch;batch.objects.resize(reads.size());
    if(!op.admitted){for(auto &r:batch.objects)r.status=Status::busy;return batch;}
    if(reads.empty() || reads.size()>8 || numa<0)return batch;
    until=i.deadline(until);
    struct Plan {
        Allocation allocation;size_t disk=0;bool claimed=false,selected=false;
        std::unique_ptr<Aligned> stage;std::unique_ptr<Impl::Temporary> registration;
    };
    std::vector<Plan> plans(reads.size());
    Impl::Leases leases(i);std::vector<std::unique_ptr<Impl::Temporary>> temporaries;
    Finally claims{[&]{for(auto &p:plans)if(p.claimed)i.devices[p.disk].index->unpin(p.allocation);}};
    nixl_xfer_dlist_t memory(DRAM_SEG),file(FILE_SEG);
    size_t staged_bytes=0;
    std::vector<G3Buffer> destinations;
    for(size_t n=0;n<reads.size();++n) {
        auto &r=batch.objects[n];auto &p=plans[n];const auto &query=reads[n];
        try {r.bytes=total(query.buffers);}catch(const std::invalid_argument&){continue;}
        if(std::any_of(query.buffers.begin(),query.buffers.end(),[](const G3Buffer &b){return b.type!=DRAM_SEG;}))
            continue;
        bool alias=false;
        std::vector<G3Buffer> accepted=destinations;
        for(const auto &b:query.buffers) {
            for(const auto &a:accepted)if(a.address<b.address+b.bytes && b.address<a.address+a.bytes)
                alias=true;
            accepted.push_back(b);
        }
        if(alias)continue; // scatter completion order cannot authorize overlapping destinations
        r.status=i.lease(query.buffers,leases,temporaries);
        if(r.status!=Status::success)continue;
        r.status=Status::not_found;
        for(size_t d=0;d<i.devices.size();++d) {
            auto &disk=i.devices[d];
            if(disk.index && disk.index->exists(query.key)){p.disk=d;p.selected=true;break;}
        }
    }
    for(size_t d=0;d<i.devices.size();++d) {
        std::vector<PinQuery> queries;std::vector<size_t> positions;
        for(size_t n=0;n<reads.size();++n)if(plans[n].selected && plans[n].disk==d) {
            queries.push_back({reads[n].key,reads[n].expected_allocation_id});positions.push_back(n);
        }
        if(queries.empty())continue;
        G3Metrics metrics;
        std::vector<PinResult> pinned;
        {MetricScope scope(&i,metrics,until);pinned=i.devices[d].index->pin_many(queries);}
        // A shared metadata request has one elapsed interval, attributed to its
        // leader object for scalar read compatibility and once to batch totals.
        batch.objects[positions.front()].metrics=metrics;
        batch.metrics.metadata_ns+=metrics.metadata_ns;
        batch.metrics.metadata_bytes+=metrics.metadata_bytes;
        batch.metrics.metadata_read_ns+=metrics.metadata_read_ns;
        batch.metrics.metadata_read_bytes+=metrics.metadata_read_bytes;
        batch.metrics.metadata_write_ns+=metrics.metadata_write_ns;
        batch.metrics.metadata_write_bytes+=metrics.metadata_write_bytes;
        for(size_t k=0;k<positions.size();++k) {
            auto n=positions[k];batch.objects[n].status=pinned[k].status;
            if(pinned[k].status==Status::success){plans[n].allocation=std::move(pinned[k].allocation);plans[n].claimed=true;}
        }
    }
    for(size_t n=0;n<reads.size();++n) {
        auto &r=batch.objects[n];auto &p=plans[n];const auto &query=reads[n];
        if(!p.claimed)continue;
        bool alias=false;auto accepted=destinations;
        for(const auto &b:query.buffers) {
            for(const auto &a:accepted)if(a.address<b.address+b.bytes && b.address<a.address+a.bytes)alias=true;
            accepted.push_back(b);
        }
        if(alias){r.status=Status::invalid_input;continue;}
        auto &disk=i.devices[p.disk];r.device_index=p.disk;r.identity=identity(p.allocation);
        if(r.bytes!=p.allocation.bytes){r.status=Status::invalid_input;continue;}
        r.direct=i.eligible(disk,p.allocation,query.buffers);
        auto targets=query.buffers;
        if(!r.direct) {
            const auto bytes=p.allocation.slots.size()*disk.index->unit_bytes();
            if(bytes>i.config.staging_bytes-staged_bytes){r.status=Status::no_space;continue;}
            staged_bytes+=bytes;
            p.stage=std::make_unique<Aligned>(4096,bytes);
            p.registration=std::make_unique<Impl::Temporary>(i,p.stage->pointer,bytes);
            targets={{reinterpret_cast<uintptr_t>(p.stage->pointer),bytes,DRAM_SEG,0,0}};
        }
        i.descriptors(disk,p.allocation,targets,memory,file);
        destinations=std::move(accepted);
    }
    if(memory.descCount()) {
        const auto start=Clock::now();const auto result=i.transfer(false,memory,file,until);
        batch.metrics.payload_ns=ns(start);
        for(size_t n=0;n<reads.size();++n) {
            auto &r=batch.objects[n];auto &p=plans[n];
            if(r.status!=Status::success)continue;
            r.status=result; // one shared request failure conservatively fails all participating objects
            if(result==Status::success) {
                r.metrics.payload_bytes=p.allocation.slots.size()*i.devices[p.disk].index->unit_bytes();
                batch.metrics.payload_bytes+=r.metrics.payload_bytes;
            }
            if(result==Status::success && p.stage) {
                const auto copy_start=Clock::now();
                size_t pos=0;
                for(const auto &b:reads[n].buffers) {
                    std::memcpy(reinterpret_cast<void*>(b.address),static_cast<char*>(p.stage->pointer)+pos,b.bytes);
                    pos+=b.bytes;
                }
                r.metrics.copy_ns=ns(copy_start);batch.metrics.copy_ns+=r.metrics.copy_ns;
                r.metrics.copy_bytes=r.bytes;batch.metrics.copy_bytes+=r.bytes;
            }
            i.fail_payload(p.disk,result);
        }
    }
    return batch;
}
G3Result G3TransferLayer::read(const std::string&key,const std::vector<G3Buffer>&buffers,
                              int numa,G3Deadline until,uint64_t expected) {
    auto batch=read_batch({{key,buffers,expected}},numa,until);
    auto result=batch.objects.front();result.metrics.payload_ns=batch.metrics.payload_ns;return result;
}
G3Result G3TransferLayer::write(const std::string&key,const std::vector<G3Buffer>&buffers,G3Deadline until) {
    return write(key,buffers,infer_numa(buffers),until);
}
G3Result G3TransferLayer::read(const std::string&key,const std::vector<G3Buffer>&buffers,
                              G3Deadline until,uint64_t expected) {
    return read(key,buffers,infer_numa(buffers),until,expected);
}
bool G3TransferLayer::exists(const std::string&key)const {
    Impl::Operation op(*impl_,false);if(!op.admitted)return false;
    for(const auto &disk:impl_->devices)if(disk.index && disk.index->exists(key))return true;
    return false;
}
std::vector<G3Entry> G3TransferLayer::enumerate()const {
    std::vector<G3Entry> out;
    Impl::Operation op(*impl_,false);if(!op.admitted)return out;
    for(size_t d=0;d<impl_->devices.size();++d)if(impl_->devices[d].index)
        for(const auto &entry:impl_->devices[d].index->enumerate())out.push_back({d,entry});
    return out;
}
std::vector<G3DeviceInfo> G3TransferLayer::devices()const {
    std::vector<G3DeviceInfo> out;
    Impl::Operation op(*impl_,false);
    for(size_t d=0;d<impl_->devices.size();++d) {
        const auto &disk=impl_->devices[d];
        out.push_back({d,disk.config.disk.path,disk.numa,
                       op.admitted&&disk.index?disk.index->state():DeviceState::closed,
                       op.admitted&&disk.index?disk.index->generation():0,disk.error});
    }
    return out;
}
Status G3TransferLayer::set_device_state(size_t d,DeviceState state) {
    Impl::Operation op(*impl_);if(!op.admitted)return Status::not_ready;
    if(d>=impl_->devices.size() || !impl_->devices[d].index)return Status::invalid_input;
    return impl_->devices[d].index->set_state(state);
}
Status G3TransferLayer::checkpoint() {
    auto &i=*impl_;Impl::Operation op(i);if(!op.admitted)return Status::not_ready;
    Status result=Status::success;
    for(auto &disk:i.devices)if(disk.index) {
        auto status=disk.index->checkpoint();if(status!=Status::success)result=status;
    }
    return result;
}
G3CloseResult G3TransferLayer::close(CloseMode mode) {
    auto &i=*impl_;std::lock_guard close_guard(i.close_mutex);
    {std::unique_lock lock(i.mutex);if(i.closed)return i.close_result;
     i.stopping=true;i.idle.wait(lock,[&]{return i.active==0;});}
    if(!i.close_recorded) {
        i.close_result={};i.close_result.clean=mode==CloseMode::clean;
        for(size_t d=0;d<i.devices.size();++d) {
            auto result=i.devices[d].index?i.devices[d].index->close(mode):Status::io_error;
            i.close_result.devices.push_back({d,result});
            i.close_result.clean&=result==Status::success;
        }
        i.close_recorded=true; // a retry cannot upgrade a DISCARD or hide an earlier failure
    }
    std::vector<uint64_t> registrations;
    {std::lock_guard lock(i.mutex);for(const auto &[id,r]:i.registrations){(void)r;registrations.push_back(id);}}
    for(auto id:registrations)if(deregister_memory(id)!=Status::success)i.close_result.clean=false;
    bool released=true;
    for(size_t d=0;d<i.devices.size();++d)if(i.devices[d].index) {
        auto status=i.retire_file(i.devices[d].index->fd());
        if(status==Status::success)i.devices[d].index.reset();
        else {
            released=false;i.close_result.clean=false;
            for(auto &r:i.close_result.devices)if(r.device_index==d)r.status=status;
        }
    }
    {std::lock_guard lock(i.mutex);i.closed=released&&i.registrations.empty();}
    return i.close_result;
}
int G3TransferLayer::infer_numa(const std::vector<G3Buffer>&buffers) {
    total(buffers);
    const auto page=static_cast<uintptr_t>(::sysconf(_SC_PAGESIZE));int chosen=-1;
    for(const auto &buffer:buffers) {
        if(buffer.type!=DRAM_SEG)throw std::invalid_argument("VRAM NUMA inference unsupported by POSIX context");
        uintptr_t current=buffer.address&~(page-1),last=(buffer.address+buffer.bytes-1)&~(page-1);
        for(;;) {
            void *address=reinterpret_cast<void*>(current);int node=-1;
            if(::syscall(SYS_move_pages,0,1,&address,nullptr,&node,0)<0 || node<0)
                throw std::invalid_argument("buffer NUMA topology unavailable; supply explicit affinity");
            if(chosen>=0 && chosen!=node)throw std::invalid_argument("buffer spans mixed NUMA nodes");
            chosen=node;if(current==last)break;current+=page;
        }
    }
    return chosen;
}

namespace {
class G3OpenQueue {
    std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<std::function<void()>> queue_;
    std::vector<std::thread> workers_;
    bool stopping_=false;
public:
    G3OpenQueue() {
        try {
        for(unsigned n=0;n<2;++n)workers_.emplace_back([this]{
            for(;;) {
                std::function<void()> fn;
                {std::unique_lock lock(mutex_);wake_.wait(lock,[&]{return stopping_||!queue_.empty();});
                 if(queue_.empty()){if(stopping_)return;continue;}
                 fn=std::move(queue_.front());queue_.pop_front();}
                fn();
            }
        });
        } catch(...) {
            {std::lock_guard lock(mutex_);stopping_=true;}wake_.notify_all();
            for(auto &worker:workers_)worker.join();
            throw;
        }
    }
    ~G3OpenQueue() {
        {std::lock_guard lock(mutex_);stopping_=true;}wake_.notify_all();
        for(auto &worker:workers_)worker.join();
    }
    bool submit(std::function<void()> fn) {
        std::lock_guard lock(mutex_);if(stopping_ || queue_.size()>=64)return false;
        queue_.push_back(std::move(fn));wake_.notify_one();return true;
    }
};
}
struct G3Session::Impl {
    std::unique_ptr<G3TransferLayer> layer;
    mutable std::mutex mutex;
    std::condition_variable wake;
    std::deque<std::function<void()>> queue;
    std::vector<std::thread> workers;
    size_t capacity,active=0;
    uint64_t timeout_ms;
    bool closing=false,close_running=false,stopped=false;
    CloseMode mode=CloseMode::discard;
    std::promise<G3CloseResult> close_promise;
    std::shared_future<G3CloseResult> close_future=close_promise.get_future().share();
    Impl(std::unique_ptr<G3TransferLayer> l,size_t count,size_t cap,uint64_t timeout)
        :layer(std::move(l)),capacity(cap),timeout_ms(timeout) {
        try {
            for(size_t n=0;n<count;++n)workers.emplace_back([this]{worker();});
        } catch(...) {
            {std::lock_guard lock(mutex);stopped=true;}wake.notify_all();
            for(auto &w:workers)w.join();
            throw;
        }
    }
    ~Impl() {
        request_close(CloseMode::discard).wait();
        for(auto &w:workers)if(w.joinable())w.join();
    }
    G3Deadline deadline(G3Deadline until) const {
        return std::min(until,Clock::now()+std::chrono::milliseconds(timeout_ms));
    }
    template<class T,class Fn> std::future<T> submit(Fn fn,std::vector<G3Buffer> buffers={},
                                                    G3MemoryHandle retire=0) {
        std::unique_lock lock(mutex);
        {
            if(closing || stopped || queue.size()>=capacity) {
                std::promise<T> reject;auto result=reject.get_future();
                reject.set_exception(std::make_exception_ptr(std::runtime_error(
                    closing?"G3 session is closing":"G3 session queue capacity exceeded")));
                return result;
            }
        }
        bool retired=false;
        try {
            auto guard=buffers.empty()?std::shared_ptr<void>{}:layer->acquire_async_use(buffers);
            if(retire) {
                if(layer->start_async_deregister(retire)!=Status::success)
                    throw std::runtime_error("G3 registration missing or already retiring");
                retired=true;
            }
            auto task=std::make_shared<std::packaged_task<T()>>(
                [this,fn=std::move(fn),guard=std::move(guard)]()mutable{
                    try {
                        std::optional<T> result;
                        layer->run_with_async_use(guard,[&]{result.emplace(fn());});
                        guard.reset(); // future retention must not retain operation leases
                        return std::move(*result);
                    } catch(...) {guard.reset();throw;}
                });
            auto future=task->get_future();queue.push_back([task]{(*task)();});
            lock.unlock();wake.notify_one();return future;
        } catch(...) {
            if(retired)layer->cancel_async_deregister(retire);
            std::promise<T> reject;auto result=reject.get_future();
            reject.set_exception(std::current_exception());return result;
        }
    }
    std::shared_future<G3CloseResult> request_close(CloseMode requested) {
        {std::lock_guard lock(mutex);if(!closing){closing=true;mode=requested;}}
        wake.notify_all();return close_future;
    }
    void worker() {
        for(;;) {
            std::function<void()> fn;bool close=false;
            {
                std::unique_lock lock(mutex);
                wake.wait(lock,[&]{return stopped||!queue.empty()||(closing&&!active&&!close_running);});
                if(stopped)return;
                if(!queue.empty()){fn=std::move(queue.front());queue.pop_front();++active;}
                else {close_running=true;close=true;}
            }
            if(close) {
                try{close_promise.set_value(layer->close(mode));}
                catch(...){close_promise.set_exception(std::current_exception());}
                {std::lock_guard lock(mutex);stopped=true;}wake.notify_all();return;
            }
            fn();
            {std::lock_guard lock(mutex);--active;}wake.notify_all();
        }
    }
};
G3Session::G3Session(std::unique_ptr<G3TransferLayer> layer,size_t workers,size_t capacity,uint64_t timeout)
    :impl_(std::make_unique<Impl>(std::move(layer),workers,capacity,timeout)) {}
G3Session::~G3Session()=default;
std::future<std::shared_ptr<G3Session>> G3Session::open(G3Config config,G3Context context,
                                                     size_t workers,size_t capacity) {
    auto task=std::make_shared<std::packaged_task<std::shared_ptr<G3Session>()>>(
        [config=std::move(config),context=std::move(context),workers,capacity]()mutable{
            if(!workers || workers>64 || !capacity || capacity>4096)
                throw std::invalid_argument("invalid G3 session worker/queue bounds");
            auto timeout=config.timeout_ms;
            auto layer=std::make_unique<G3TransferLayer>(std::move(config),std::move(context));
            return std::shared_ptr<G3Session>(new G3Session(std::move(layer),workers,capacity,timeout));
        });
    auto future=task->get_future();static G3OpenQueue open_queue;
    if(!open_queue.submit([task]{(*task)();})) {
        std::promise<std::shared_ptr<G3Session>> rejected;auto result=rejected.get_future();
        rejected.set_exception(std::make_exception_ptr(std::runtime_error("G3 open queue capacity exceeded")));
        return result;
    }
    return future;
}
std::future<G3MemoryHandle> G3Session::register_memory(std::vector<G3Buffer> buffers) {
    auto *i=impl_.get();return i->submit<G3MemoryHandle>([i,buffers=std::move(buffers)]{
        return i->layer->register_memory(buffers);
    });
}
std::future<G3MemoryHandle> G3Session::borrow_registered_memory(std::vector<G3Buffer> buffers) {
    auto *i=impl_.get();return i->submit<G3MemoryHandle>([i,buffers=std::move(buffers)]{
        return i->layer->borrow_registered_memory(buffers);
    });
}
std::future<Status> G3Session::deregister_memory(G3MemoryHandle handle) {
    auto *i=impl_.get();return i->submit<Status>([i,handle]{return i->layer->finish_deregister(handle,true);},{},handle);
}
std::future<G3Result> G3Session::write(std::string key,std::vector<G3Buffer> buffers,int numa,G3Deadline until) {
    auto *i=impl_.get();until=i->deadline(until);auto admission=buffers;
    return i->submit<G3Result>([i,key=std::move(key),buffers=std::move(buffers),numa,until]{
        return i->layer->write(key,buffers,numa,until);
    },std::move(admission));
}
std::future<G3Result> G3Session::write(std::string key,std::vector<G3Buffer> buffers,G3Deadline until) {
    auto *i=impl_.get();until=i->deadline(until);auto admission=buffers;
    return i->submit<G3Result>([i,key=std::move(key),buffers=std::move(buffers),until]{
        return i->layer->write(key,buffers,until);
    },std::move(admission));
}
std::future<G3Result> G3Session::read(std::string key,std::vector<G3Buffer> buffers,int numa,
                                   G3Deadline until,uint64_t expected) {
    auto *i=impl_.get();until=i->deadline(until);auto admission=buffers;
    return i->submit<G3Result>([i,key=std::move(key),buffers=std::move(buffers),numa,until,expected]{
        return i->layer->read(key,buffers,numa,until,expected);
    },std::move(admission));
}
std::future<G3Result> G3Session::read(std::string key,std::vector<G3Buffer> buffers,
                                   G3Deadline until,uint64_t expected) {
    auto *i=impl_.get();until=i->deadline(until);auto admission=buffers;
    return i->submit<G3Result>([i,key=std::move(key),buffers=std::move(buffers),until,expected]{
        return i->layer->read(key,buffers,until,expected);
    },std::move(admission));
}
std::future<G3BatchResult> G3Session::read_batch(std::vector<G3Read> reads,int numa,G3Deadline until) {
    auto *i=impl_.get();until=i->deadline(until);
    std::vector<G3Buffer> admission;
    for(const auto &r:reads)admission.insert(admission.end(),r.buffers.begin(),r.buffers.end());
    return i->submit<G3BatchResult>([i,reads=std::move(reads),numa,until]{
        return i->layer->read_batch(reads,numa,until);
    },std::move(admission));
}
bool G3Session::exists(const std::string &key)const{return impl_->layer->exists(key);}
std::shared_future<G3CloseResult> G3Session::close(CloseMode mode){return impl_->request_close(mode);}
} // namespace nixlshard
