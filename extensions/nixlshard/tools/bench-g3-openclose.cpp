// Standalone authoritative G3 measurements, synthetic payloads only.
#include "nixlshard/g3.h"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
using namespace nixlshard;
using Clock=std::chrono::steady_clock;
uint64_t ns(Clock::time_point a){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-a).count();}
void check(bool a,const char*m){if(!a)throw std::runtime_error(m);}
struct Memory {void*p=nullptr;size_t n;Memory(size_t z):n(z){check(posix_memalign(&p,4096,n)==0,"aligned allocation");}~Memory(){free(p);}};
int main(int argc,char**argv) {
 try {
 check(argc==3,"usage: benchmark assigned-debug-file results-json");
 constexpr size_t page=16UL<<20,half=8UL<<20;
 nixlAgent sdk("g3-open-close",nixlAgentConfig(false,false,0,nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT));
 nixlBackendH*backend=nullptr;check(sdk.createBackend("POSIX",{{"use_aio","true"}},backend)==NIXL_SUCCESS,"POSIX");
 G3Context context;context.agent=&sdk;context.agent_name="g3-open-close";context.file_options.backends={backend};context.memory_options=context.file_options;
 DiskConfig disk;disk.path=argv[1];disk.capacity_bytes=128UL<<30;disk.unit_bytes=page;disk.create=true;disk.direct_io=true;disk.min_object_bytes=disk.max_object_bytes=page;disk.key_bytes=32;disk.metadata_alignment=4096;disk.namespace_id="synthetic-authoritative-g3-128-pages-v1";
 G3Config config;config.instance_id="standalone-benchmark";config.devices={{disk,0}};config.memory_mode=MemoryMode::explicit_registration;config.staging_bytes=128UL<<20;config.timeout_ms=5000;
 Memory source(page),destination(8*page);memset(source.p,0x3a,source.n);
 std::vector<AllocationIdentity> ids;std::ofstream out(argv[2]);out<<"{\"storage_contract\":\"authoritative_g3_v2\",\"page_bytes\":"<<page<<",\"key_bytes\":32,\"unit_bytes\":"<<page<<",\"objects\":128,\"capacity_bytes\":"<<disk.capacity_bytes<<",\"numa_node\":0,\"memory_mode\":\"EXPLICIT\",\"direct_io\":true,\"samples\":[";out.flush();
 for(int cycle=0;cycle<5;++cycle) {
  auto begin=Clock::now();G3TransferLayer layer(config,context);auto open_ns=ns(begin);
  auto s=layer.register_memory({{reinterpret_cast<uintptr_t>(source.p),source.n}});
  auto d=layer.register_memory({{reinterpret_cast<uintptr_t>(destination.p),destination.n}});
  if(cycle==0)for(int n=0;n<128;++n) {
   auto w=layer.write(std::string(32,char(n+1)),{{reinterpret_cast<uintptr_t>(source.p),page,DRAM_SEG,0,s}},0);
   check(w.status==Status::success&&w.direct&&w.metrics.copy_bytes==0,"seed direct write");ids.push_back(w.identity);
  }
  uint64_t payload_ns=0,metadata_ns=0,payload_bytes=0,metadata_bytes=0,copy_bytes=0,wall_ns=0;
  for(int n=0;n<128;n+=8) {
   memset(destination.p,0,destination.n);std::vector<G3Read> reads;
   for(int j=0;j<8;++j)reads.push_back({std::string(32,char(n+j+1)),
    {{reinterpret_cast<uintptr_t>(destination.p)+j*page,half,DRAM_SEG,0,d},
     {reinterpret_cast<uintptr_t>(destination.p)+j*page+half,half,DRAM_SEG,0,d}},ids[n+j].id});
   begin=Clock::now();auto batch=layer.read_batch(reads,0);wall_ns+=ns(begin);
   for(const auto&r:batch.objects)check(r.status==Status::success&&r.direct,"batch direct read");
   check(!memcmp(source.p,destination.p,page),"complete page0 value");
   for(int j=1;j<8;++j)check(!memcmp(source.p,static_cast<char*>(destination.p)+j*page,page),"complete other value");
   payload_ns+=batch.metrics.payload_ns;metadata_ns+=batch.metrics.metadata_ns;
   payload_bytes+=batch.metrics.payload_bytes;metadata_bytes+=batch.metrics.metadata_bytes;copy_bytes+=batch.metrics.copy_bytes;
  }
  check(payload_bytes==128*page&&copy_bytes==0&&metadata_bytes>0,"exact completed bytes/metadata/direct copy proof");
  check(layer.deregister_memory(s)==Status::success&&layer.deregister_memory(d)==Status::success,"registration retirement");
  begin=Clock::now();auto closed=layer.close(CloseMode::clean);auto close_ns=ns(begin);check(closed.clean,"CLEAN close");
  if(cycle)out<<",";out<<"{\"repeat\":"<<cycle<<",\"open_ns\":"<<open_ns<<",\"clean_close_ns\":"<<close_ns<<",\"read_batch_wall_ns\":"<<wall_ns<<",\"payload_ns\":"<<payload_ns<<",\"metadata_ns\":"<<metadata_ns<<",\"payload_bytes\":"<<payload_bytes<<",\"metadata_bytes\":"<<metadata_bytes<<",\"copy_bytes\":"<<copy_bytes<<",\"groups\":16,\"pages_per_group\":8,\"destination_segments_per_page\":2,\"correct\":true}";
  out.flush();std::cout<<"cycle "<<cycle<<" open_ns "<<open_ns<<" clean_close_ns "<<close_ns<<" payload_bytes "<<payload_bytes<<" metadata_bytes "<<metadata_bytes<<" copy_bytes "<<copy_bytes<<" PASS"<<std::endl;
 }
 out<<"],\"passed\":true,\"scope\":\"Standalone G3 open and CLEAN close wall time; five 128-page replays using one native payload request per 8-page group. Value comparisons occur outside read wall timing. First open initializes new format; later opens recover same CLEAN media. Synthetic payloads, no SGLang TTFT claim.\"}\n";
 return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}
}
