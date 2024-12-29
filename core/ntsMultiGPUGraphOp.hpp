
#ifndef NTSMULTIGPUGRAPGOP_HPP
#define NTSMULTIGPUGRAPGOP_HPP
#include <assert.h>
#include <map>
#include <math.h>
#include <stack>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>

#include "core/graph.hpp"
#include "core/ntsBaseOp.hpp"
#include "core/DeviceSubStructure.hpp"

namespace nts {
namespace op {

#if CUDA_ENABLE    
class MultiGPUAllGNNCalcGraphSumOp:public ntsGraphOp{
  
  DeviceSubStructure *subgraph;
public:

  MultiGPUAllGNNCalcGraphSumOp(DeviceSubStructure *subgraph_)
  {
    this->subgraph = subgraph_;
  }
  NtsVar forward(NtsVar &f_input)
  {
    // LOG_INFO("device[%d]start graph forward for epoch[%d]\n", subgraph->device_id, subgraph->graph->rtminfo->epoch);
    int feature_size = f_input.size(1);

    NtsVar f_output = subgraph->graph->Nts->NewKeyTensor({subgraph->owned_vertices, feature_size}, 
                                                        torch::DeviceType::CUDA, subgraph->device_id);
    
    ValueType *f_input_buffer =
      subgraph->graph->Nts->getWritableBuffer(f_input, torch::DeviceType::CUDA);
    ValueType *f_output_buffer =
      subgraph->graph->Nts->getWritableBuffer(f_output, torch::DeviceType::CUDA);

    subgraph->sync_and_compute(f_input_buffer, feature_size, f_output_buffer);

    // std::ofstream out("./log/cora_aggregate_" + std::to_string(subgraph->device_id) + ".txt", std::ios_base::out);//for debug
    // // cudaPointerAttributes attributes;
    // // CHECK_CUDA_RESULT(cudaPointerGetAttributes(&attributes, f_output));
    // // out << "(out data)GPU ID: " << attributes.device << std::endl;
    // out << "(out data)GPU ID: " << f_output.device() << std::endl;
    // out << "(out data)size: " << f_output.sizes() << std::endl;
    // // NtsVar f_output_cpu = f_output.to(torch::DeviceType::CPU);
    // ValueType *f_output_cpu = (ValueType *)malloc(sizeof(ValueType) * subgraph->owned_vertices * feature_size);
    // CHECK_CUDA_RESULT(cudaMemcpy(f_output_cpu, f_output_buffer, sizeof(ValueType) * subgraph->owned_vertices * feature_size,
    //                     cudaMemcpyDeviceToHost));
    // for(int i = 0; i < subgraph->owned_vertices; i++)
    // {
    //   out << "dst id[" << i + subgraph->subgraph_offset[subgraph->device_id] << "] embedding: ";
    //   for(int j = 0; j < feature_size; j++)
    //   {
    //     // out << f_output_cpu[i].data<float>()[j] << " ";
    //     out << f_output_cpu[i * feature_size + j] << " ";
    //   }
    //   out << std::endl;
    // }
    // sleep(5);
    // std::cout << "finish no bug!!!!!!!!!!" << std::endl;
    return f_output;
  }

  NtsVar backward(NtsVar &f_output_grad)
  {
    int feature_size = f_output_grad.size(1);

    NtsVar f_input_grad = subgraph->graph->Nts->NewKeyTensor({subgraph->owned_vertices, feature_size}, 
                                                        torch::DeviceType::CUDA, subgraph->device_id);

    ValueType *f_output_buffer =
      subgraph->graph->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CUDA);
    ValueType *f_input_buffer =
      subgraph->graph->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CUDA);

    subgraph->compute_and_sync_backward(f_output_buffer, feature_size, f_input_buffer);

    return f_input_grad;

  }
};

class MultiGPUAllGNNCalcGraphSumOpSpMM:public ntsGraphOp{
  
  DeviceSubStructure *subgraph;
public:

  MultiGPUAllGNNCalcGraphSumOpSpMM(DeviceSubStructure *subgraph_)
  {
    this->subgraph = subgraph_;
  }
  NtsVar forward(NtsVar &f_input)
  {
    int feature_size = f_input.size(1);

    NtsVar f_output = subgraph->graph->Nts->NewKeyTensor({subgraph->owned_vertices, feature_size}, 
                                                        torch::DeviceType::CUDA, subgraph->device_id);
    
    ValueType *f_input_buffer =
      subgraph->graph->Nts->getWritableBuffer(f_input, torch::DeviceType::CUDA);
    ValueType *f_output_buffer =
      subgraph->graph->Nts->getWritableBuffer(f_output, torch::DeviceType::CUDA);

    subgraph->sync_and_compute_SpMM(f_input_buffer, feature_size, f_output_buffer);

    return f_output;
  }

  NtsVar backward(NtsVar &f_output_grad)
  {
    int feature_size = f_output_grad.size(1);

    NtsVar f_input_grad = subgraph->graph->Nts->NewKeyTensor({subgraph->owned_vertices, feature_size}, 
                                                        torch::DeviceType::CUDA, subgraph->device_id);

    ValueType *f_output_buffer =
      subgraph->graph->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CUDA);
    ValueType *f_input_buffer =
      subgraph->graph->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CUDA);

    // subgraph->compute_and_sync_backward(f_output_buffer, feature_size, f_input_buffer);
    subgraph->compute_and_sync_backward_SpMM(f_output_buffer, feature_size, f_input_buffer);

    return f_input_grad;

  }
};


class MultiGPUAllGNNCalcGraphSumOpSpMM_WithBcast:public ntsGraphOp{
  
  DeviceSubStructure *subgraph;
public:

  MultiGPUAllGNNCalcGraphSumOpSpMM_WithBcast(DeviceSubStructure *subgraph_)
  {
    this->subgraph = subgraph_;
  }
  NtsVar forward(NtsVar &f_input)
  {
    int feature_size = f_input.size(1);

    NtsVar f_output = subgraph->graph->Nts->NewKeyTensor({subgraph->owned_vertices, feature_size}, 
                                                        torch::DeviceType::CUDA, subgraph->device_id);
    
    ValueType *f_input_buffer =
      subgraph->graph->Nts->getWritableBuffer(f_input, torch::DeviceType::CUDA);
    ValueType *f_output_buffer =
      subgraph->graph->Nts->getWritableBuffer(f_output, torch::DeviceType::CUDA);

    subgraph->bcast_and_compute_SpMM(f_input_buffer, feature_size, f_output_buffer);

    return f_output;
  }

  NtsVar backward(NtsVar &f_output_grad)
  {
    int feature_size = f_output_grad.size(1);

    NtsVar f_input_grad = subgraph->graph->Nts->NewKeyTensor({subgraph->owned_vertices, feature_size}, 
                                                        torch::DeviceType::CUDA, subgraph->device_id);

    ValueType *f_output_buffer =
      subgraph->graph->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CUDA);
    ValueType *f_input_buffer =
      subgraph->graph->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CUDA);

    subgraph->compute_and_sync_backward_SpMM(f_output_buffer, feature_size, f_input_buffer);

    return f_input_grad;

  }
};



} // namespace graphop
} // namespace nts

#endif //end CUDA_ENABLE

#endif //end head define
