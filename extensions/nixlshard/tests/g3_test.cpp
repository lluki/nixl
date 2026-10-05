/* SPDX-License-Identifier: Apache-2.0 */
#include "nixlshard/g3.h"
#include <atomic>
#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <future>
#include <iostream>
#include <libaio.h>
#include <mutex>
#include <stdexcept>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
using namespace nixlshard;
using namespace std::chrono_literals;
namespace {
std::atomic<bool> block_io=false,entered_io=false,fail_barrier=false;
void check(bool condition,const char *message){if(!condition)throw std::runtime_error(message);}
struct Buffer {
 void *p=nullptr;size_t bytes;
 explicit Buffer(size_t n):bytes(n){if(posix_memalign(&p,4096,n))throw std::bad_alloc();std::memset(p,0,n);}
 ~Buffer(){std::free(p);}
 std::vector<G3Buffer> whole(){return {{reinterpret_cast<uintptr_t>(p),bytes}};}
};
struct Fixture {
 std::filesystem::path dir;
 nixlAgent sdk;
 nixlBackendH *backend=nullptr;
 G3Context context;
 Fixture():dir(std::filesystem::temp_directory_path()/("nixlshard-g3-"+std::to_string(getpid()))),
    sdk("g3-test",nixlAgentConfig(false,false,0,nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT)) {
  std::filesystem::create_directories(dir);
  check(sdk.createBackend("POSIX",{{"use_aio","true"}},backend)==NIXL_SUCCESS,"POSIX backend");
  context.agent=&sdk;context.agent_name="g3-test";context.file_options.backends={backend};
  context.memory_options=context.file_options;
 }
 ~Fixture(){std::filesystem::remove_all(dir);}
 G3Config config(const std::string &name,MemoryMode mode=MemoryMode::automatic) {
  DiskConfig disk;disk.path=(dir/name).string();disk.capacity_bytes=16*1024*1024;
  disk.unit_bytes=4096;disk.create=true;disk.direct_io=true;disk.min_object_bytes=100;
  disk.max_object_bytes=8192;disk.key_bytes=32;disk.namespace_id="exact-model/layout";
  G3Config cfg;cfg.instance_id="g3-test-instance";cfg.devices={{disk,0}};
  cfg.memory_mode=mode;cfg.staging_bytes=65536;cfg.max_active=4;return cfg;
 }
};
std::string key(unsigned value){return std::string(32,static_cast<char>(value));}
void automatic_batch_and_recovery(Fixture &f) {
 auto cfg=f.config("automatic.bin");
 Buffer source(8192),target(8192*8);std::memset(source.p,0x51,8192);
 std::vector<AllocationIdentity> identities;
 {
  G3TransferLayer layer(cfg,f.context);
  for(unsigned n=1;n<=8;++n) {
   auto w=layer.write(key(n),source.whole(),0);
   check(w.status==Status::success && w.direct && w.metrics.copy_bytes==0,"automatic direct write");
   check(w.metrics.metadata_bytes>0 && w.metrics.payload_bytes==8192,"write metadata/payload accounting");
   identities.push_back(w.identity);
  }
  std::vector<G3Read> reads;
  for(unsigned n=1;n<=8;++n)reads.push_back({key(n),{{reinterpret_cast<uintptr_t>(target.p)+(n-1)*8192,4096},
        {reinterpret_cast<uintptr_t>(target.p)+(n-1)*8192+4096,4096}},identities[n-1].id});
  auto read=layer.read_batch(reads,0);
  check(read.objects.size()==8 && read.metrics.payload_bytes==65536 &&
        read.metrics.metadata_bytes>=8*4096 && read.metrics.payload_ns>0,"batched authoritative metadata and payload");
  for(auto &r:read.objects)check(r.status==Status::success && r.direct && r.metrics.copy_bytes==0,"batched status");
  for(size_t n=0;n<target.bytes;++n)check(static_cast<unsigned char*>(target.p)[n]==0x51,"scatter bytes");
  auto miss=layer.read(key(1),source.whole(),0,G3Deadline::max(),identities[0].id+100);
  check(miss.status==Status::not_found,"stale identity rejected");
  check(layer.write(key(99),source.whole(),1).status==Status::invalid_input,"strict local placement");
  check(layer.enumerate().size()==8,"enumeration");
  check(layer.checkpoint()==Status::success,"DIRTY checkpoint");
  auto close=layer.close(CloseMode::clean);check(close.clean && close.devices.size()==1,"clean close");
 }
 {
  G3TransferLayer layer(cfg,f.context);
  check(layer.enumerate().size()==8 && layer.exists(key(1)),"clean recovery");
  check(layer.read(key(1),source.whole(),0).status==Status::success,"recovered read");
  layer.close(CloseMode::discard);
 }
 {
  G3TransferLayer layer(cfg,f.context);
  check(!layer.exists(key(1)) && layer.enumerate().empty(),"dirty reset");
  layer.close(CloseMode::discard);
  check(!layer.close(CloseMode::clean).clean,"discard cannot upgrade");
 }
 std::cout<<"automatic batch, metadata reads, clean restore/dirty reset PASS\n";
}
void explicit_and_padding(Fixture &f) {
 auto cfg=f.config("explicit.bin",MemoryMode::explicit_registration);
 Buffer memory(8192);std::memset(memory.p,0x27,memory.bytes);
 G3TransferLayer layer(cfg,f.context);
 check(layer.write(key(10),memory.whole(),0).status==Status::invalid_input,"EXPLICIT rejects undeclared buffer");
 nixl_reg_dlist_t descriptor(DRAM_SEG);descriptor.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(memory.p),memory.bytes,0,""));
 check(f.sdk.registerMem(descriptor,&f.context.memory_options)==NIXL_SUCCESS,"external registration");
 auto borrowed=layer.borrow_registered_memory(memory.whole());
 auto input=memory.whole();input[0].bytes=100;input[0].registration=borrowed;
 auto result=layer.write(key(10),input,0);
 check(result.status==Status::success && !result.direct && result.metrics.copy_bytes==100 &&
       result.metrics.copy_ns>0 && result.metrics.payload_bytes==4096,"owned aligned padding");
 std::memset(memory.p,0,8192);
 auto read=layer.read(key(10),input,0);
 check(read.status==Status::success && !read.direct && read.metrics.copy_bytes==100 &&
       static_cast<unsigned char*>(memory.p)[99]==0x27 &&
       static_cast<unsigned char*>(memory.p)[100]==0,"no caller overrun");
 check(layer.deregister_memory(borrowed)==Status::success,"borrowed declaration retirement");
 check(f.sdk.deregisterMem(descriptor,&f.context.memory_options)==NIXL_SUCCESS,"borrow did not deregister external owner");
 auto owned=layer.register_memory(memory.whole());
 check(layer.write(key(11),memory.whole(),0).status==Status::success,"owned registration");
 check(layer.deregister_memory(owned)==Status::success,"owned retirement");
 check(layer.write(key(12),{{1,8192,VRAM_SEG,0}},0).status==Status::invalid_input,"unsupported VRAM never CPU dereferenced");
 bool unknown=false;try{G3TransferLayer::infer_numa({{1,8192}});}catch(const std::invalid_argument&){unknown=true;}
 check(unknown,"unknown kernel affinity explicit error");
 layer.close(CloseMode::discard);
 std::cout<<"explicit owned/borrowed registration and bounded padding PASS\n";
}
void startup_and_failures(Fixture &f) {
 auto cfg=f.config("partial.bin");
 auto bad=cfg.devices.front();bad.disk.path=(f.dir/"nonexistent"/"bad.bin").string();
 cfg.devices.insert(cfg.devices.begin(),bad);
 G3TransferLayer layer(cfg,f.context);
 auto info=layer.devices();check(info.size()==2 && !info[0].diagnostic.empty() &&
   info[1].state==DeviceState::active,"partial startup diagnostics");
 Buffer source(8192);check(layer.write(key(22),source.whole(),0).status==Status::success,"healthy disk remains");
 auto close=layer.close(CloseMode::clean);
 check(close.devices.size()==2 && close.devices[0].status!=Status::success && !close.clean,"partial close results explicit");
 auto debug=f.config("unknown-numa.bin");debug.devices[0].numa_node=-1;
 bool rejected=false;try{G3TransferLayer invalid(debug,f.context);}catch(const std::runtime_error&){rejected=true;}
 check(rejected,"debug topology override required");
 auto barrier=f.config("barrier.bin");G3TransferLayer failing(barrier,f.context);
 check(failing.write(key(30),source.whole(),0).status==Status::success,"barrier write");
 fail_barrier=true;auto failure=failing.close(CloseMode::clean);fail_barrier=false;
 check(!failure.clean && failure.devices[0].status==Status::io_error,"clean barrier failure visible");
 std::cout<<"partial startup, strict topology, close failures PASS\n";
}
void registration_drain(Fixture &f) {
 auto cfg=f.config("drain.bin",MemoryMode::explicit_registration);
 G3TransferLayer layer(cfg,f.context);Buffer buffer(8192);
 auto handle=layer.register_memory(buffer.whole());
 check(layer.write(key(31),buffer.whole(),0).status==Status::success,"drain seed");
 entered_io=false;block_io=true;
 auto read=std::async(std::launch::async,[&]{return layer.read(key(31),buffer.whole(),0);});
 auto end=std::chrono::steady_clock::now()+5s;
 while(!entered_io && std::chrono::steady_clock::now()<end)std::this_thread::sleep_for(1ms);
 if(!entered_io){block_io=false;read.wait();throw std::runtime_error("LinuxAIO interposition not observed");}
 auto retire=std::async(std::launch::async,[&]{return layer.deregister_memory(handle);});
 check(retire.wait_for(30ms)==std::future_status::timeout,"deregister waits reference drain");
 auto rejected=layer.read(key(31),buffer.whole(),0);
 check(rejected.status==Status::not_ready,"retired registration blocks new use");
 block_io=false;
 check(read.get().status==Status::success && retire.get()==Status::success,"live registration quiescent release");
 layer.close(CloseMode::discard);
 std::cout<<"actual NIXL LinuxAIO registration drain PASS\n";
}
}
namespace {
void instance_identity_and_direction(Fixture &f) {
 auto cfg=f.config("identity-a.bin");auto second=cfg.devices.front();
 second.disk.path=(f.dir/"identity-b.bin").string();cfg.devices.push_back(second);
 Buffer source(8192),target(8192);std::memset(source.p,0x73,source.bytes);
 G3TransferLayer layer(cfg,f.context);
 check(layer.set_device_state(0,DeviceState::draining)==Status::success,"first disk draining");
 auto first=layer.write(key(50),source.whole(),0);
 check(first.status==Status::success && first.device_index==1,"healthy later assigned disk placement");
 check(layer.set_device_state(0,DeviceState::active)==Status::success,"first disk active");
 std::memset(source.p,0x44,source.bytes);auto duplicate=layer.write(key(50),source.whole(),0);
 check(duplicate.status==Status::success && duplicate.device_index==1 &&
       duplicate.identity.id==first.identity.id && duplicate.metrics.payload_bytes==0 &&
       duplicate.metrics.metadata_read_bytes>0 && duplicate.metrics.metadata_write_bytes==0 &&
       layer.enumerate().size()==1,"immutable instance key cannot duplicate onto earlier free disk");
 auto read=layer.read(key(50),target.whole(),1);
 check(read.status==Status::success && static_cast<unsigned char*>(target.p)[0]==0x73,
       "read affinity does not move recorded placement or change immutable value");
 auto short_buffer=source.whole();short_buffer[0].bytes=4096;
 check(layer.write(key(50),short_buffer,0).status==Status::invalid_input,"existing length checked");
 auto batch=layer.read_batch({{key(50),target.whole()},{key(50),target.whole()}},0);
 check(batch.objects[0].status==Status::success && batch.objects[1].status==Status::invalid_input,
       "overlapping batch destinations explicitly rejected");
 auto failed=layer.read(key(50),target.whole(),0,std::chrono::steady_clock::now()-1ms);
 check(failed.status==Status::timeout && failed.metrics.payload_bytes==0 &&
       failed.metrics.metadata_bytes==0,"deadline failures do not count completed transfer bytes");
 layer.close(CloseMode::discard);
 std::cout<<"instance-wide immutable identity, direction counters, deadlines and alias rejection PASS\n";
}

void async_registration_admission(Fixture &f) {
 auto cfg=f.config("async-registration.bin",MemoryMode::explicit_registration);
 auto session=G3Session::open(cfg,f.context,2,8).get();
 Buffer source(8192),target(8192),second(8192);
 std::memset(source.p,0x76,source.bytes);
 auto source_handle=session->register_memory(source.whole()).get();
 auto target_handle=session->register_memory(target.whole()).get();
 auto second_handle=session->register_memory(second.whole()).get();
 check(session->write(key(46),source.whole(),0).get().status==Status::success,"async explicit seed");
 entered_io=false;block_io=true;
 auto first=session->read(key(46),target.whole(),0);
 auto end=std::chrono::steady_clock::now()+5s;
 while(!entered_io && std::chrono::steady_clock::now()<end)std::this_thread::sleep_for(1ms);
 if(!entered_io){block_io=false;first.wait();throw std::runtime_error("async registration AIO interposition missing");}
 auto accepted=session->read(key(46),second.whole(),0);
 auto retired=session->deregister_memory(second_handle);
 auto late=session->read(key(46),second.whole(),0);
 bool rejected=false;try{late.get();}catch(const std::runtime_error&){rejected=true;}
 check(rejected,"deregister submission rejects newer queued use");
 check(retired.wait_for(30ms)==std::future_status::timeout,"deregister drains accepted queued use");
 block_io=false;
 check(first.get().status==Status::success && accepted.get().status==Status::success,
       "retirement preserves older accepted registration leases");
 check(retired.get()==Status::success,"deregister future releases admission leases");
 check(!std::memcmp(source.p,second.p,source.bytes),"accepted queued read complete data");
 check(session->deregister_memory(source_handle).get()==Status::success &&
       session->deregister_memory(target_handle).get()==Status::success,"remaining async registrations retire");
 check(session->close(CloseMode::clean).get().clean,"async registration clean close");
 std::cout<<"async admission leases/retire/new-use rejection PASS\n";
}
void bounded_async(Fixture &f) {
 auto cfg=f.config("session.bin");Buffer source(8192),target(8192),second(8192);
 auto session=G3Session::open(cfg,f.context,1,1).get();
 check(session->write(key(45),source.whole(),0).get().status==Status::success,"async write completion");
 entered_io=false;block_io=true;
 auto first=session->read(key(45),target.whole(),0);
 auto end=std::chrono::steady_clock::now()+5s;
 while(!entered_io && std::chrono::steady_clock::now()<end)std::this_thread::sleep_for(1ms);
 if(!entered_io){block_io=false;first.wait();throw std::runtime_error("async AIO interposition missing");}
 auto queued=session->read(key(45),second.whole(),0);
 auto overflow=session->read(key(45),source.whole(),0);
 bool refused=false;try{overflow.get();}catch(const std::runtime_error&){refused=true;}
 check(refused,"fixed pending queue rejects overflow");
 auto close=session->close(CloseMode::clean);
 check(close.wait_for(30ms)==std::future_status::timeout,"async close waits accepted work");
 auto late=session->read(key(45),source.whole(),0);
 refused=false;try{late.get();}catch(const std::runtime_error&){refused=true;}
 check(refused,"async close stops new admissions");
 block_io=false;
 check(first.get().status==Status::success && queued.get().status==Status::success,
       "async close drains queued and active reads");
 check(close.get().clean && !session->exists(key(45)),"async clean completion");
 std::cout<<"bounded async open/read/write/close completion PASS\n";
}
}
extern "C" int io_submit(io_context_t context,long count,struct iocb **blocks) {
 if(block_io){entered_io=true;while(block_io)std::this_thread::sleep_for(1ms);}
 const auto result=syscall(SYS_io_submit,context,count,blocks);
 return result<0?-errno:static_cast<int>(result);
}
extern "C" int fdatasync(int fd) {
 if(fail_barrier){errno=EIO;return -1;}
 return syscall(SYS_fdatasync,fd);
}
int main() {
 try {
  Fixture fixture;
  automatic_batch_and_recovery(fixture);
  explicit_and_padding(fixture);
  startup_and_failures(fixture);
  registration_drain(fixture);
  instance_identity_and_direction(fixture);
  async_registration_admission(fixture);
  bounded_async(fixture);
  return 0;
 } catch(const std::exception &e) {
  block_io=false;std::cerr<<"G3 test: "<<e.what()<<"\n";return 1;
 }
}
