#include "core/neutronstar.hpp"

class GCN_DP_impl {
public:
    int iterations;

    
    ValueType learn_rate;
    ValueType weight_decay;
    ValueType drop_rate;
    ValueType alpha;
    ValueType beta1;
    ValueType beta2;
    ValueType epsilon;
    ValueType decay_rate;
    ValueType decay_epoch;

    int num_devices;//num_device = 0是CPU执行
    int num_pipelines;
    int K;
    ValueType appnp_alpha;
    
    // graph
    VertexSubset *active;
    // graph with no edge data
    Graph<Empty> *graph;
    FullyRepGraph* full_rep_graph;

    MultiDeviceGenerator* subgraph_generator; 
    
    //NN
    GNNDatum *gnndatum;
    // NtsVar label_cpu;
    // NtsVar label_gpu;
    // NtsVar MASK;

    // Variables
    std::vector<Parameter *> P;
    std::vector<NtsVar> X;
    nts::ctx::NtsContext* ctx;
    std::vector<nts::ctx::NtsContext *> device_ctx;//for multiGPU
    
    NtsVar F;
    // NtsVar loss;
    std::vector<ValueType> train_acc;
    std::vector<ValueType> val_acc;
    std::vector<ValueType> test_acc;
    ValueType max_train_acc = 0;
    ValueType max_val_acc = 0;
    ValueType max_test_acc = 0;

    std::vector<unsigned long> train_correct;
    std::vector<unsigned long> val_correct;
    std::vector<unsigned long> test_correct;
    std::vector<double> per_epoch_time;
    double all_epoch_time = 0.0;


    
    GCN_DP_impl(Graph<Empty> *graph_, int iterations_,
               bool process_local = false, bool process_overlap = false){
        graph = graph_;
        iterations = iterations_;

        LOG_INFO("config gpu num: %d", graph->config->gpu_num);
        num_devices = graph->config->gpu_num;
        if(num_devices == -1) {
            cudaGetDeviceCount(&num_devices);
        }
        assert(num_devices >= 0);

        LOG_INFO("config pipeline num: %d", graph->config->pipeline_num);
        num_pipelines = graph->config->pipeline_num;
        if(num_pipelines <= 0) {
            num_pipelines = 3;
        }
        assert(num_pipelines >= 1);

        active = graph->alloc_vertex_subset();
        active->fill();
        graph->init_gnnctx(graph->config->layer_string);
        //APPNP不采样
        // graph->init_gnnctx_fanout(graph->config->fanout_string);
        // rtminfo initialize
        graph->init_rtminfo();
        graph->rtminfo->process_local = graph->config->process_local;
        graph->rtminfo->reduce_comm = graph->config->process_local;
        graph->rtminfo->copy_data = false;
        graph->rtminfo->process_overlap = graph->config->overlap;
        graph->rtminfo->with_weight = true;
        graph->rtminfo->lock_free = graph->config->lock_free;
        appnp_alpha = graph->config->alpha;
        K = graph->config->K;
    }

    void init_graph(){
        full_rep_graph = new FullyRepGraph(graph);
        full_rep_graph->GenerateAll();
        full_rep_graph->SyncAndLog("read_finish");
        if(num_devices > 1){//多GPU要创建多个ctx，还没写

        } else {
            ctx=new nts::ctx::NtsContext();
        }
    }

    void init_nn(){
        learn_rate = graph->config->learn_rate;
        weight_decay = graph->config->weight_decay;
        drop_rate = graph->config->drop_rate;
        alpha = graph->config->learn_rate;
        decay_rate = graph->config->decay_rate;
        decay_epoch = graph->config->decay_epoch;
        beta1 = 0.9;
        beta2 = 0.999;
        epsilon = 1e-9;

        gnndatum = new GNNDatum(graph->gnnctx, graph);
        if (0 == graph->config->feature_file.compare("random")) {
            gnndatum->random_generate();
        } else {
            gnndatum->readFeature_Label_Mask(graph->config->feature_file,
                                             graph->config->label_file,
                                             graph->config->mask_file);
        }
        // gnndatum->registLabel(label_cpu);
        // gnndatum->registMask(MASK);

        for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
            P.push_back(new Parameter(graph->gnnctx->layer_size[i],
                                        graph->gnnctx->layer_size[i + 1], alpha, beta1,
                                        beta2, epsilon, weight_decay));
        }

        for (int i = 0; i < P.size(); i++) {
            P[i]->init_parameter();
            P[i]->set_decay(decay_rate, decay_epoch);
        }

        // F = graph->Nts->NewLeafTensor( //
        //     gnndatum->local_feature,
        //     {graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]},
        //     torch::DeviceType::CPU);

        if(num_devices == 0){
            graph->rtminfo->with_cuda = false;
            init_CPU_version();
        }
        else {
            graph->rtminfo->with_cuda = true;
            if(num_devices > 1){
                init_multiGPU_version();
            } else {
                init_singleGPU_version();
            }
        }

        // debug_init_info();
    }

    void init_CPU_version(){  
        subgraph_generator = new MultiDeviceGenerator(full_rep_graph, gnndatum, num_devices);
        subgraph_generator->CPU_version();
    }

    void init_singleGPU_version(){

    }
  
    void init_multiGPU_version(){
        train_correct.resize(num_devices, 0);
        val_correct.resize(num_devices, 0);
        test_correct.resize(num_devices, 0);

        subgraph_generator = new MultiDeviceGenerator(full_rep_graph, gnndatum, num_devices);
        subgraph_generator->multi_GPU_version();

        std::vector<int> arr(num_devices);
        std::iota(arr.begin(), arr.end(), 0);
        NCCL_Communicator *nccl_comm = new NCCL_Communicator(num_devices, arr.data());

        for(int i = 0; i < num_devices; i++) {
            for(int j = 0; j < graph->gnnctx->layer_size.size() - 1; j++) {
                subgraph_generator->subgraph_queq[i]->P.push_back(new Parameter(graph->gnnctx->layer_size[j], graph->gnnctx->layer_size[j + 1], 
                                                                                alpha, beta1, beta2, epsilon, weight_decay));
                subgraph_generator->subgraph_queq[i]->P[j]->W.set_data(P[j]->W);
                subgraph_generator->subgraph_queq[i]->P[j]->set_decay(decay_rate, decay_epoch);

                subgraph_generator->subgraph_queq[i]->P[j]->set_multi_gpu_comm(nccl_comm);
            }
        }

        // subgraph_generator->init_CudaStream();//放到每个子图的定义里面了
        // NCCL_Communicator *nccl_graph_comm = new NCCL_Communicator(num_devices);
        // subgraph_generator->init_communicator(nccl_graph_comm);
        subgraph_generator->init_communicator(nccl_comm);
        subgraph_generator->load_data_to_corresponding_device();

    }

    NtsVar LOSS(NtsVar &predict, NtsVar &label, nts::ctx::NtsContext* ctx, NtsVar &mask)
    {
        //应该改成只对train点算loss
        // auto loss = torch::nll_loss(predict, label);
        // if(ctx->training == true)
        // {
        //     ctx->appendNNOp(predict, loss);
        // }
        // return loss;

        torch::Tensor a = predict.log_softmax(-1);
        NtsVar mask_train = mask.eq(0);
        auto loss_train = torch::nll_loss(
            a.masked_select(mask_train.expand({mask_train.size(0), a.size(1)}))
                   .view({-1, a.size(1)}),
            label.masked_select(mask_train.view({mask_train.size(0)}))
        );
        if(ctx->training == true)
        {
            ctx->appendNNOp(predict, loss_train);
        }
        // LOG_INFO("loss:%f", loss_train.item<float>());

        return loss_train;
    }

    long getCorrect(long s, NtsVar &predict, NtsVar &label, NtsVar &mask) { // 0 train, //1 eval //2 test
        NtsVar mask_train = mask.eq(s);
        NtsVar all_train =
            predict
                .argmax(1)
                .to(torch::kLong)
                .eq(label)
                .to(torch::kLong)
                .masked_select(mask_train.view({mask_train.size(0)}));
        return  all_train.sum(0).item<long>();
    }

    void Update(DeviceSubStructure *subgraph)
    {
        for(int i = 0; i < subgraph->P.size(); i++)
        {
            subgraph->P[i]->reduce_multi_gpu_gradient(subgraph->P[i]->W.grad(), subgraph->device_id, subgraph->cs->stream);
            subgraph->P[i]->reset_gradient();
            subgraph->P[i]->learnG_with_decay_Adam();
            subgraph->P[i]->next();
        }
        for (int i = 0; i < subgraph->P.size(); i++) {
           subgraph->P[i]->zero_grad();
        }
        // sleep(5);
        // assert(false);
    }
    
    void acc(int type, int epoch, bool log = false)
    {
        unsigned correct_num = 0;
        if(type == 0){
            for(auto&& num : train_correct)
            {
                correct_num += num;
            }
            ValueType acc = 1.0 * correct_num / subgraph_generator->train_num;
            max_train_acc = max_train_acc > acc ? max_train_acc : acc;
            if(log){
                std::cout << "GNNmini::Running.Epoch[" << epoch << "]:Times["
                    << per_epoch_time[epoch] << "(s)]" << std::endl;
                LOG_INFO("Train Acc: %f %d %d", acc, correct_num, subgraph_generator->train_num);
            } else {
                train_acc.push_back(acc);
            }
        } else if(type == 1){
            for(auto&& num : val_correct)
            {
                correct_num += num;
            }
            ValueType acc = 1.0 * correct_num / subgraph_generator->val_num;
            max_val_acc = max_val_acc > acc ? max_val_acc : acc;
            if(log){
                LOG_INFO("val Acc: %f %d %d", acc, correct_num, subgraph_generator->val_num);
            } else {
                val_acc.push_back(acc);
            }
        } else{
            for(auto&& num : test_correct)
            {
                correct_num += num;
            }
            ValueType acc = 1.0 * correct_num / subgraph_generator->test_num;
            max_test_acc = max_test_acc > acc ? max_test_acc : acc;
            if(log){
                LOG_INFO("test Acc: %f %d %d", acc, correct_num, subgraph_generator->test_num);
            } else {
                test_acc.push_back(acc);
            }
        }


    }

    void CPU_Version()
    {
        LOG_INFO("GNNmini::[Single.CPU.GCNimpl] running [%d] Epoches\n",
               iterations);
        for (int i = 0; i < graph->gnnctx->layer_size.size(); i++) {
            NtsVar d;
            X.push_back(d);
        }
        X[0] = subgraph_generator->subgraph_queq[0]->feature.set_requires_grad(true);
        
        for (int i_i = 0; i_i < iterations; i_i++) 
        {
            graph->rtminfo->epoch = i_i;
            if (i_i != 0) {
                for (int i = 0; i < P.size(); i++) {
                    P[i]->zero_grad();  
                }
            }

            //forward
            graph->rtminfo->forward = true;
            for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
            graph->rtminfo->curr_layer = i;

            
            // NtsVar Y_i= ctx->runGraphOp<nts::op::ForwardCPUfuseOp>(subgraph_generator->subgraph_queq[0],active,X[i]);      
            //     X[i + 1]=ctx->runVertexForward([&](NtsVar n_i,NtsVar v_i){
            //         return vertexForward(n_i, v_i);
            //     },
            //     Y_i,
            //     X[i]);
            }
        }
        
    }

    void Multi_Forward(DeviceSubStructure *subgraph)
    {
        // std::ofstream out("./log/cora_GCNGPUs_" + std::to_string(subgraph->device_id) + ".txt", std::ios_base::out);//for debug
        
        graph->rtminfo->forward = true;
        VertexId dev_id = subgraph->device_id;

        train_correct[dev_id] = 0;
        val_correct[dev_id] = 0;
        test_correct[dev_id] = 0;
        
        at::cuda::setCurrentCUDAStream(*(subgraph->ts));
        cudaSetUsingDevice(dev_id);
        for(int layer = 0; layer < (graph->gnnctx->layer_size.size()-1); layer++)
        {
            // out << "-------------------------------------------layer :" << layer << std::endl;
            graph->rtminfo->curr_layer = layer;

                // out << "layer[" << layer << "] X sum: " << subgraph->X[layer].abs().sum() << std::endl;
                // out << "layer[" << layer << "] X mean: " << subgraph->X[layer].abs().mean() << std::endl;
                // out << "layer[" << layer << "] X max: " << subgraph->X[layer].abs().max() << std::endl;

            NtsVar Y_i = device_ctx[dev_id]->runGraphOp<nts::op::MultiGPUAllGNNCalcGraphSumOp>(subgraph, subgraph->X[layer]);
            // NtsVar Y_i = device_ctx[dev_id]->runGraphOp<nts::op::MultiGPUAllGNNCalcGraphSumOpSpMM>(subgraph, subgraph->X[layer]);

                // out << "layer[" << layer << "] Y_i sum: " << Y_i.abs().sum() << std::endl;
                // out << "layer[" << layer << "] Y_i mean: " << Y_i.abs().mean() << std::endl;
                // out << "layer[" << layer << "] Y_i max: " << Y_i.abs().max() << std::endl;
                
            subgraph->X[layer+1] = device_ctx[dev_id]->runVertexForward([&](NtsVar n_i, NtsVar v_i){
                if(layer < (graph->gnnctx->layer_size.size()-2)){
                    return torch::dropout(torch::relu(subgraph->P[layer]->forward(n_i)), drop_rate, device_ctx[dev_id]->is_train());
                }
                else{
                    return subgraph->P[layer]->forward(n_i);
                    }
            }, Y_i, subgraph->X[layer]);

                // out << "layer[" << layer << "] subgraph->P[layer]->W sum : " << subgraph->P[layer]->W.abs().sum() << std::endl;
                // out << "layer[" << layer << "] subgraph->P[layer]->W mean : " << subgraph->P[layer]->W.abs().mean() << std::endl;
                // out << "layer[" << layer << "] subgraph->P[layer]->W max : " << subgraph->P[layer]->W.abs().max() << std::endl;
                

                // out << "layer[" << layer << "] X+1: " << subgraph->X[layer+1].abs().sum() << std::endl;
                // out << "layer[" << layer << "] X+1 mean: " << subgraph->X[layer+1].abs().mean() << std::endl;
                // out << "layer[" << layer << "] X+1 max: " << subgraph->X[layer+1].abs().max() << std::endl;
                // out << "finish NN OP" << std::endl;
                // out << "(out data)GPU ID: " << subgraph->X[layer+1].device() << std::endl;
                // out << "(out data)size: " << subgraph->X[layer+1].sizes() << std::endl;
                // NtsVar X_i_1 = subgraph->X[layer+1].to(torch::DeviceType::CPU);
                // for(int i = 0; i < subgraph->owned_vertices; i++)
                // {
                //     out << "dst id[" << i + subgraph->subgraph_offset[subgraph->device_id] << "] embedding: ";
                //     for(int j = 0; j < X_i_1.size(1); j++)
                //     {
                //         out << X_i_1[i].data<float>()[j] << " ";
                //     }
                //     out << std::endl;
                // }
        }
        // sleep(5);
        // assert(false);

        NtsVar loss = LOSS(subgraph->X[graph->gnnctx->layer_size.size()-1], subgraph->label_gpu, device_ctx[dev_id], 
                            subgraph->mask_gpu);

                // out << "loss sum : " << loss.abs().sum() << std::endl;
                // out << "loss mean: " << loss.abs().mean() << std::endl;
                // out << "loss max: " << loss.abs().max() << std::endl;
        
        cudaStreamSynchronize(subgraph->cs->stream);
        
        train_correct[dev_id] = getCorrect(0, subgraph->X[graph->gnnctx->layer_size.size()-1], 
                                            subgraph->label_gpu, subgraph->mask_gpu);
        val_correct[dev_id] = getCorrect(1, subgraph->X[graph->gnnctx->layer_size.size()-1], 
                                            subgraph->label_gpu, subgraph->mask_gpu);
        test_correct[dev_id] = getCorrect(2, subgraph->X[graph->gnnctx->layer_size.size()-1], 
                                            subgraph->label_gpu, subgraph->mask_gpu);

            // out << "finish LOSS OP" << std::endl;
            // out << "(label)GPU ID: " << subgraph->label_gpu.device() << std::endl;
            // out << "(label)size: " << subgraph->label_gpu.sizes() << std::endl;
            // out << "(mask)GPU ID: " << subgraph->mask_gpu.device() << std::endl;
            // out << "(mask)size: " << subgraph->mask_gpu.sizes() << std::endl;
            // out << "(loss) : " << loss << std::endl;
            // out << "(train correct) : " << train_correct[dev_id] << std::endl;
            // out << "(val correct) : " << val_correct[dev_id] << std::endl;
            // out << "(test correct) : " << test_correct[dev_id] << std::endl;
            // NtsVar lab = subgraph->label_gpu.to(torch::DeviceType::CPU);
            // NtsVar mask = subgraph->mask_gpu.to(torch::DeviceType::CPU);
            // for(int i = 0; i < lab.size(0); i++)
            // {
            //     out << "dst id[" << i + subgraph->subgraph_offset[subgraph->device_id] << "] label/mask: ";
            //     out << lab.data<long>()[i]<< " " << mask.data<int>()[i]<< " ";
            //     out << std::endl;
            // }

        // device_ctx[dev_id]->self_backward(out, false);
        // sleep(5);
        // assert(false);
        device_ctx[dev_id]->self_backward(false);

        Update(subgraph);

        //Updata for debug
        // for(int i = 0; i < subgraph->P.size(); i++)
        // {
        //     // out << "layer[" << i << "] W grad size: " << subgraph->P[i]->W.grad().sizes() << std::endl;
        //     // out << "layer[" << i << "] W grad device: " << subgraph->P[i]->W.grad().device() << std::endl;
        //     // out << "layer[" << i << "] before allreduce W grad size: " << subgraph->P[i]->W << std::endl;
        //     subgraph->P[i]->reduce_multi_gpu_gradient(subgraph->P[i]->W.grad(), subgraph->device_id, subgraph->cs->stream);
        //     // out << "layer[" << i << "] after allreduce W grad size: " << subgraph->P[i]->W << std::endl;
        //     subgraph->P[i]->learnC2G_with_decay_Adam();
        //     // subgraph->P[i]->learn_local_with_decay_Adam();
        //     subgraph->P[i]->next();
        // }
        // for (int i = 0; i < subgraph->P.size(); i++) {
        //    subgraph->P[i]->zero_grad();
        // }


    }
    
    void Multi_GPU_Version()
    {
        LOG_INFO("GNNmini::[MultiGPU[%d].CPU.GCNimpl] running [%d] Epoches\n", num_devices, iterations);
        device_ctx.resize(num_devices);
        for(int i = 0; i < num_devices; i++) {
            device_ctx[i] = new nts::ctx::NtsContext();
        }

        for (int i_i = 0; i_i < iterations; i_i++) 
        {
            double epoch_time = 0.0;
            epoch_time -= get_time();

            graph->rtminfo->epoch = i_i;
            if (i_i != 0) {
                for(int i = 0; i < num_devices; i++) {
                    for(int j = 0; j < graph->gnnctx->layer_size.size() - 1; j++) {
                        subgraph_generator->subgraph_queq[i]->P[j]->zero_grad();   
                    }
                }
            }

            for(int device_id = 0; device_id < num_devices; device_id++) {
                device_ctx[device_id]->train();
            }

            //forward
            std::vector<std::thread> device_threads;
            for(int device_id = 0; device_id < num_devices; device_id++)
            {
                device_threads.emplace_back([&](int dev_id){                    
                    assert(dev_id == subgraph_generator->subgraph_queq[dev_id]->device_id);
                    Multi_Forward(subgraph_generator->subgraph_queq[dev_id]);
                }, device_id);
            }
            for(int device_id = 0; device_id < num_devices; device_id++) {
                device_threads[device_id].join();
            }

            epoch_time += get_time();
            per_epoch_time.push_back(epoch_time);

            acc(0, i_i, true);
            acc(1, i_i, true);
            acc(2, i_i, true);

        }

        for(auto&& t : per_epoch_time)
        {
            all_epoch_time += t;
        }
        std::cout << "--------------------finish algorithm !" << std::endl;
        std::cout << "all_epoch_time: " << all_epoch_time << std::endl;
        std::cout << "max train acc: " << max_train_acc<< std::endl;
        std::cout << "max val acc: " << max_val_acc<< std::endl;
        std::cout << "max test acc: " << max_test_acc<< std::endl;
        

    }

    void run(){
        if(num_devices == 0)
        {
            CPU_Version();
        } else if (num_devices == 1)
        {
            
        } else
        {
            Multi_GPU_Version();
        }
        

    }

    void debug_init_info()
    {
        for(int i = 0; i < full_rep_graph->global_vertices; i++)
        {
            printf("dst[%d]: src[", i);
            for(int j = full_rep_graph->column_offset[i]; j < full_rep_graph->column_offset[i+1]; j++)
            {
                printf("%d ", full_rep_graph->row_indices[j]);
            }
            std::cout << "]" << std::endl;
        }
    }


};