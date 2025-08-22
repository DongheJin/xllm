#include "collective_service.h"

#include <brpc/closure_guard.h>
#include <glog/logging.h>

#include <vector>

namespace xllm {

CollectiveService::CollectiveService(int dp_group_num,
                                     int total_num,
                                     int device_idx)
    : total_num_(total_num) {
  root_infos_.reserve(dp_group_num + 1);
  for (size_t i = 0; i < (dp_group_num + 1); ++i) {
    HcclRootInfo root_info;
    auto error = aclrtSetDevice(device_idx);
    CHECK_EQ(error, ACL_SUCCESS)
        << "ACL set device id " << device_idx << " failed. Error : " << error;
    auto status = HcclGetRootInfo(&root_info);
    CHECK_EQ(status, HCCL_SUCCESS) << "HCCL get root info failed.";
    root_infos_.push_back(root_info);
  }
}

void CollectiveService::Sync(::google::protobuf::RpcController* controller,
                             const proto::AddressInfo* request,
                             proto::CommUniqueIdList* response,
                             ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);

  std::string address = request->address();
  int32_t global_rank = request->global_rank();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    addrs_map_[global_rank] = address;
  }

  to_proto_list(root_infos_, response);

  return;
}

std::unordered_map<int32_t, std::string> CollectiveService::wait() {
  int connected = 0;
  while (connected < total_num_) {
    absl::SleepFor(absl::Milliseconds(1000));
    {
      std::lock_guard<std::mutex> lock(mutex_);
      connected = addrs_map_.size();
    }
  }

  return addrs_map_;
}

void CollectiveService::to_proto_list(const std::vector<HcclRootInfo>& src,
                                      proto::CommUniqueIdList* dst) {
  for (const auto& id : src) {
    proto::CommUniqueId* protoId = dst->add_comm_unique_ids();
    protoId->set_comm_unique_id(id.internal, sizeof(id.internal));
  }
}

void CollectiveService::from_proto_list(const proto::CommUniqueIdList& src,
                                        std::vector<HcclRootInfo>* dst) {
  dst->clear();
  for (const auto& protoId : src.comm_unique_ids()) {
    HcclRootInfo id;
    std::memcpy(
        id.internal, protoId.comm_unique_id().data(), sizeof(id.internal));
    dst->push_back(id);
  }
}

}  // namespace xllm
