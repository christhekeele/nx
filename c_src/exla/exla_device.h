#ifndef EXLA_DEVICE_H_
#define EXLA_DEVICE_H_

#include <memory>

#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "tensorflow/compiler/xla/pjrt/event_pool.h"
#include "tensorflow/compiler/xla/pjrt/local_device_state.h"

namespace exla {

namespace se = tensorflow::se;

// Wrapper around a single XLA device.

class ExlaDevice {
 public:
  explicit ExlaDevice(int id,
                      se::StreamExecutor* executor,
                      xla::LocalClient* client,
                      xla::LocalDeviceState::AllocationModel allocation_model,
                      bool asynchronous,
                      bool allow_event_reuse) : id_(id),
                                                client_(client) {
    auto state = std::make_unique<xla::LocalDeviceState>(executor, client,
                                                         allocation_model,
                                                         asynchronous,
                                                         allow_event_reuse);
    state_ = std::move(state);
  }

  virtual ~ExlaDevice() = default;

  const int id() const {
    return id_;
  }

  // Returns this device's device ordinal.
  const int device_ordinal() const { return state_->executor()->device_ordinal(); }

  // Returns this device's stream executor.
  se::StreamExecutor* executor() const { return state_->executor(); }

  // Returns this device's client.
  xla::LocalClient* client() const {
    return client_;
  }

  xla::EventPool& event_pool() { return state_->event_pool(); }

  // Returns this device's compute stream. Compute streams
  // are used for running computations.
  se::Stream* compute_stream() const {
    return state_->compute_stream();
  }

  // Returns this device's host-to-device stream. Host-to-device
  // streams are used for host-to-device transfers.
  se::Stream* host_to_device_stream() const {
    return state_->host_to_device_stream();
  }

  // Returns a device-to-host stream. Device-to-host
  // streams are used for device-to-host transfers.
  se::Stream* GetDeviceToHostStream() {
    return state_->GetDeviceToHostStream();
  }

  // Returns a device-to-device stream. Device-to-device
  // streams are used for device-to-device transfers.
  se::Stream* GetDeviceToDeviceStream() {
    return state_->GetDeviceToDeviceStream();
  }

  // Returns a stream from a pool. The stream is guaranteed not to have any
  // currently outstanding work at its tail.
  std::unique_ptr<se::Stream> BorrowStreamFromPool() {
    return state_->BorrowStreamFromPool();
  }

  xla::LocalDeviceState::AllocationModel allocation_model() {
    return state_->allocation_model();
  }

  // Returns a stream to the pool. The caller must ensure the stream does not
  // have any outstanding work at its tail.
  void ReturnStreamToPool(std::unique_ptr<se::Stream> stream) {
    state_->ReturnStreamToPool(std::move(stream));
  }

 private:
  const int id_;
  std::unique_ptr<xla::LocalDeviceState> state_;
  xla::LocalClient* client_;
};
}  // namespace exla

#endif
