#include <assert.h>
#include <chrono>
#include <vector>
#include <unordered_map>
#include <map>
// #include "oneapi/dnnl/dnnl.hpp"
#include "example_utils.hpp"
#include <iomanip>
#include <string.h>
// #include "utils.h"
// #include "weight_loader.h"
// #include "data_loader.h"

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;
struct tensor_dims { int N; int IC; int IH; int IW; };
class Weights
{
public:
	std::vector<float> values; 
	int64_t count;      //!< The number of weights in the array.
};

memory conv2d_onednn_wo_bias(memory &INPUT,
	std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	std::vector<float> &weights,
	tensor_dims &t_dims, int OC, int KH, int KW, int SH, int SW, int TP, int BP, int LP, int RP);

memory bn_onednn(
	memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	std::vector<float> &scale, std::vector<float> &shift, std::vector<float> &mean, std::vector<float> &var,
	tensor_dims &t_dims, float eps = 1.e-5f);

memory pooling_onednn(
	memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	tensor_dims &t_dims,int KH, int KW, int	SH, int SW, int TP, int BP, int LP, int RP,	int DH, int DW, int mode);

memory gap_onednn(
	memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	tensor_dims &t_dims);

memory fc_onednn(
	memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	std::vector<float> &weights, std::vector<float> &bias,
	tensor_dims &t_dims, int OC);

memory eltwise_onednn(
	memory &INPUT, memory &INPUT2, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream);

memory activation_onednn(
	memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	int mode = 0); // 0 : relu

void show_dims(const tensor_dims t_dims) {
	std::cout << t_dims.N << ", " << t_dims.IC << ", " << t_dims.IH << ", " << t_dims.IW << std::endl;
}


void resnet18_v2(std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &stream, std::map<std::string, Weights> &weightMap,
	int batch_size, int input_channel, int input_width, int input_height, tensor_dims &t_dims) {

	std::vector<float> inputs(batch_size * input_channel * input_height * input_width);
	//[inputs]
	auto inputs_src_md = memory::desc({ batch_size, input_channel, input_height, input_width }, dt::f32, tag::nchw);
	auto inputs_src_md_memory = memory(inputs_src_md, engine);
	write_to_dnnl_memory(inputs.data(), inputs_src_md_memory);

	// net work
	memory conv1 = conv2d_onednn_wo_bias(inputs_src_md_memory, net, net_args, engine, stream, weightMap["conv1.weight"].values, t_dims, 64, 7, 7, 2, 2, 3, 3, 3, 3);
	memory bn_relu1 = bn_onednn(conv1, net, net_args, engine, stream, weightMap["bn1.weight"].values, weightMap["bn1.bias"].values, weightMap["bn1.running_mean"].values, weightMap["bn1.running_var"].values, t_dims, 1.e-5f);
	bn_relu1 = activation_onednn(bn_relu1, net, net_args, engine, stream);
	memory pool1 = pooling_onednn(bn_relu1, net, net_args, engine, stream, t_dims, 3, 3, 2, 2, 0, 0, 1, 1, 1, 1, 1);
	//show_dims(t_dims);

	// layer1 
	// basicBlock1
	memory layer1_conv1_1 = conv2d_onednn_wo_bias(pool1, net, net_args, engine, stream, weightMap["layer1.0.conv1.weight"].values, t_dims, 64, 3, 3, 1, 1, 1, 1, 1, 1);
	memory layer1_bn_relu1_1 = bn_onednn(layer1_conv1_1, net, net_args, engine, stream, weightMap["layer1.0.bn1.weight"].values, weightMap["layer1.0.bn1.bias"].values, weightMap["layer1.0.bn1.running_mean"].values, weightMap["layer1.0.bn1.running_var"].values, t_dims, 1.e-5f);
	layer1_bn_relu1_1 = activation_onednn(layer1_bn_relu1_1, net, net_args, engine, stream);
	memory layer1_conv1_2 = conv2d_onednn_wo_bias(layer1_bn_relu1_1, net, net_args, engine, stream, weightMap["layer1.0.conv2.weight"].values, t_dims, 64, 3, 3, 1, 1, 1, 1, 1, 1);
	memory layer1_bn1_2 = bn_onednn(layer1_conv1_2, net, net_args, engine, stream, weightMap["layer1.0.bn2.weight"].values, weightMap["layer1.0.bn2.bias"].values, weightMap["layer1.0.bn2.running_mean"].values, weightMap["layer1.0.bn2.running_var"].values, t_dims, 1.e-5f);
	memory layer1_elt_sum1_3 = eltwise_onednn(pool1, layer1_bn1_2, net, net_args, engine, stream);
	memory layer1_relu1_3 = activation_onednn(layer1_elt_sum1_3, net, net_args, engine, stream);
	// basicBlock2
	memory layer1_conv1_4 = conv2d_onednn_wo_bias(layer1_relu1_3, net, net_args, engine, stream, weightMap["layer1.1.conv1.weight"].values, t_dims, 64, 3, 3, 1, 1, 1, 1, 1, 1);
	memory layer1_bn_relu1_4 = bn_onednn(layer1_conv1_4, net, net_args, engine, stream, weightMap["layer1.1.bn1.weight"].values, weightMap["layer1.1.bn1.bias"].values, weightMap["layer1.1.bn1.running_mean"].values, weightMap["layer1.1.bn1.running_var"].values, t_dims, 1.e-5f);
	layer1_bn_relu1_4 = activation_onednn(layer1_bn_relu1_4, net, net_args, engine, stream);
	memory layer1_conv1_5 = conv2d_onednn_wo_bias(layer1_bn_relu1_4, net, net_args, engine, stream, weightMap["layer1.1.conv2.weight"].values, t_dims, 64, 3, 3, 1, 1, 1, 1, 1, 1);
	memory layer1_bn5 = bn_onednn(layer1_conv1_5, net, net_args, engine, stream, weightMap["layer1.1.bn2.weight"].values, weightMap["layer1.1.bn2.bias"].values, weightMap["layer1.1.bn2.running_mean"].values, weightMap["layer1.1.bn2.running_var"].values, t_dims, 1.e-5f);
	memory layer1_elt_sum1_5 = eltwise_onednn(layer1_relu1_3, layer1_bn5, net, net_args, engine, stream);
	memory layer1_relu1_5 = activation_onednn(layer1_elt_sum1_5, net, net_args, engine, stream);
	// layer1 
	//show_dims(t_dims);

	// layer2 
	// basicBlock1
	tensor_dims t_dims2 = t_dims;
	memory layer2_conv1_1 = conv2d_onednn_wo_bias(layer1_relu1_5, net, net_args, engine, stream, weightMap["layer2.0.conv1.weight"].values, t_dims, 128, 3, 3, 2, 2, 1, 1, 1, 1);
	memory layer2_bn_relu1_1 = bn_onednn(layer2_conv1_1, net, net_args, engine, stream, weightMap["layer2.0.bn1.weight"].values, weightMap["layer2.0.bn1.bias"].values, weightMap["layer2.0.bn1.running_mean"].values, weightMap["layer2.0.bn1.running_var"].values, t_dims, 1.e-5f);
	layer2_bn_relu1_1 = activation_onednn(layer2_bn_relu1_1, net, net_args, engine, stream);
	memory layer2_conv1_2 = conv2d_onednn_wo_bias(layer2_bn_relu1_1, net, net_args, engine, stream, weightMap["layer2.0.conv2.weight"].values, t_dims, 128, 3, 3, 1, 1, 1, 1, 1, 1);
	memory layer2_bn1_2 = bn_onednn(layer2_conv1_2, net, net_args, engine, stream, weightMap["layer2.0.bn2.weight"].values, weightMap["layer2.0.bn2.bias"].values, weightMap["layer2.0.bn2.running_mean"].values, weightMap["layer2.0.bn2.running_var"].values, t_dims, 1.e-5f);
	memory layer2_down_conv1_2 = conv2d_onednn_wo_bias(layer1_relu1_5, net, net_args, engine, stream, weightMap["layer2.0.downsample.0.weight"].values, t_dims2, 128, 1, 1, 2, 2, 0, 0, 0, 0);
	memory layer2_down_bn1_2 = bn_onednn(layer2_down_conv1_2, net, net_args, engine, stream, weightMap["layer2.0.downsample.1.weight"].values, weightMap["layer2.0.downsample.1.bias"].values, weightMap["layer2.0.downsample.1.running_mean"].values, weightMap["layer2.0.downsample.1.running_var"].values, t_dims, 1.e-5f);
	memory layer2_elt_sum1_3 = eltwise_onednn(layer2_down_bn1_2, layer2_bn1_2, net, net_args, engine, stream);
	memory layer2_relu1_3 = activation_onednn(layer2_elt_sum1_3, net, net_args, engine, stream);
	// basicBlock2
	memory layer2_conv1_4 = conv2d_onednn_wo_bias(layer2_relu1_3, net, net_args, engine, stream, weightMap["layer2.1.conv1.weight"].values, t_dims, 128, 3, 3, 1, 1, 1, 1, 1, 1);
	memory layer2_bn_relu1_4 = bn_onednn(layer2_conv1_4, net, net_args, engine, stream, weightMap["layer2.1.bn1.weight"].values, weightMap["layer2.1.bn1.bias"].values, weightMap["layer2.1.bn1.running_mean"].values, weightMap["layer2.1.bn1.running_var"].values, t_dims, 1.e-5f);
	layer2_bn_relu1_4 = activation_onednn(layer2_bn_relu1_4, net, net_args, engine, stream);
	memory layer2_conv1_5 = conv2d_onednn_wo_bias(layer2_bn_relu1_4, net, net_args, engine, stream, weightMap["layer2.1.conv2.weight"].values, t_dims, 128, 3, 3, 1, 1, 1, 1, 1, 1);
	memory layer2_bn5 = bn_onednn(layer2_conv1_5, net, net_args, engine, stream, weightMap["layer2.1.bn2.weight"].values, weightMap["layer2.1.bn2.bias"].values, weightMap["layer2.1.bn2.running_mean"].values, weightMap["layer2.1.bn2.running_var"].values, t_dims, 1.e-5f);
	memory layer2_elt_sum1_5 = eltwise_onednn(layer2_relu1_3, layer2_bn5, net, net_args, engine, stream);
	memory layer2_relu1_5 = activation_onednn(layer2_elt_sum1_5, net, net_args, engine, stream);
	// layer2 
	//show_dims(t_dims);

	// layer3 
	// basicBlock1
	t_dims2 = t_dims;
	memory layer3_conv1_1 = conv2d_onednn_wo_bias(layer2_relu1_5, net, net_args, engine, stream, weightMap["layer3.0.conv1.weight"].values, t_dims, 256, 3, 3, 2, 2, 1, 1, 1, 1);
	memory layer3_bn_relu1_1 = bn_onednn(layer3_conv1_1, net, net_args, engine, stream, weightMap["layer3.0.bn1.weight"].values, weightMap["layer3.0.bn1.bias"].values, weightMap["layer3.0.bn1.running_mean"].values, weightMap["layer3.0.bn1.running_var"].values, t_dims, 1.e-5f);
	layer3_bn_relu1_1 = activation_onednn(layer3_bn_relu1_1, net, net_args, engine, stream);
	memory layer3_conv1_2 = conv2d_onednn_wo_bias(layer3_bn_relu1_1, net, net_args, engine, stream, weightMap["layer3.0.conv2.weight"].values, t_dims, 256, 3, 3, 1, 1, 1, 1, 1, 1);
	memory layer3_bn1_2 = bn_onednn(layer3_conv1_2, net, net_args, engine, stream, weightMap["layer3.0.bn2.weight"].values, weightMap["layer3.0.bn2.bias"].values, weightMap["layer3.0.bn2.running_mean"].values, weightMap["layer3.0.bn2.running_var"].values, t_dims, 1.e-5f);
	memory layer3_down_conv1_2 = conv2d_onednn_wo_bias(layer2_relu1_5, net, net_args, engine, stream, weightMap["layer3.0.downsample.0.weight"].values, t_dims2, 256, 1, 1, 2, 2, 0, 0, 0, 0);
	memory layer3_down_bn1_2 = bn_onednn(layer3_down_conv1_2, net, net_args, engine, stream, weightMap["layer3.0.downsample.1.weight"].values, weightMap["layer3.0.downsample.1.bias"].values, weightMap["layer3.0.downsample.1.running_mean"].values, weightMap["layer3.0.downsample.1.running_var"].values, t_dims, 1.e-5f);
	memory layer3_elt_sum1_3 = eltwise_onednn(layer3_down_bn1_2, layer3_bn1_2, net, net_args, engine, stream);
	memory layer3_relu1_3 = activation_onednn(layer3_elt_sum1_3, net, net_args, engine, stream);
	// basicBlock2
	memory layer3_conv1_4 = conv2d_onednn_wo_bias(layer3_relu1_3, net, net_args, engine, stream, weightMap["layer3.1.conv1.weight"].values, t_dims, 256, 3, 3, 1, 1, 1, 1, 1, 1);
	memory layer3_bn_relu1_4 = bn_onednn(layer3_conv1_4, net, net_args, engine, stream, weightMap["layer3.1.bn1.weight"].values, weightMap["layer3.1.bn1.bias"].values, weightMap["layer3.1.bn1.running_mean"].values, weightMap["layer3.1.bn1.running_var"].values, t_dims, 1.e-5f);
	layer3_bn_relu1_4 = activation_onednn(layer3_bn_relu1_4, net, net_args, engine, stream);
	memory layer3_conv1_5 = conv2d_onednn_wo_bias(layer3_bn_relu1_4, net, net_args, engine, stream, weightMap["layer3.1.conv2.weight"].values, t_dims, 256, 3, 3, 1, 1, 1, 1, 1, 1);
	memory layer3_bn5 = bn_onednn(layer3_conv1_5, net, net_args, engine, stream, weightMap["layer3.1.bn2.weight"].values, weightMap["layer3.1.bn2.bias"].values, weightMap["layer3.1.bn2.running_mean"].values, weightMap["layer3.1.bn2.running_var"].values, t_dims, 1.e-5f);
	memory layer3_elt_sum1_5 = eltwise_onednn(layer3_relu1_3, layer3_bn5, net, net_args, engine, stream);
	memory layer3_relu1_5 = activation_onednn(layer3_elt_sum1_5, net, net_args, engine, stream);
	// layer3 
	//show_dims(t_dims);

	// layer4 
	// basicBlock1
	t_dims2 = t_dims;
	memory layer4_conv1_1 = conv2d_onednn_wo_bias(layer3_relu1_5, net, net_args, engine, stream, weightMap["layer4.0.conv1.weight"].values, t_dims, 512, 3, 3, 2, 2, 1, 1, 1, 1);
	memory layer4_bn_relu1_1 = bn_onednn(layer4_conv1_1, net, net_args, engine, stream, weightMap["layer4.0.bn1.weight"].values, weightMap["layer4.0.bn1.bias"].values, weightMap["layer4.0.bn1.running_mean"].values, weightMap["layer4.0.bn1.running_var"].values, t_dims, 1.e-5f);
	layer4_bn_relu1_1 = activation_onednn(layer4_bn_relu1_1, net, net_args, engine, stream);
	memory layer4_conv1_2 = conv2d_onednn_wo_bias(layer4_bn_relu1_1, net, net_args, engine, stream, weightMap["layer4.0.conv2.weight"].values, t_dims, 512, 3, 3, 1, 1, 1, 1, 1, 1);
	memory layer4_bn1_2 = bn_onednn(layer4_conv1_2, net, net_args, engine, stream, weightMap["layer4.0.bn2.weight"].values, weightMap["layer4.0.bn2.bias"].values, weightMap["layer4.0.bn2.running_mean"].values, weightMap["layer4.0.bn2.running_var"].values, t_dims, 1.e-5f);
	memory layer4_down_conv1_2 = conv2d_onednn_wo_bias(layer3_relu1_5, net, net_args, engine, stream, weightMap["layer4.0.downsample.0.weight"].values, t_dims2, 512, 1, 1, 2, 2, 0, 0, 0, 0);
	memory layer4_down_bn1_2 = bn_onednn(layer4_down_conv1_2, net, net_args, engine, stream, weightMap["layer4.0.downsample.1.weight"].values, weightMap["layer4.0.downsample.1.bias"].values, weightMap["layer4.0.downsample.1.running_mean"].values, weightMap["layer4.0.downsample.1.running_var"].values, t_dims, 1.e-5f);
	memory layer4_elt_sum1_3 = eltwise_onednn(layer4_down_bn1_2, layer4_bn1_2, net, net_args, engine, stream);
	memory layer4_relu1_3 = activation_onednn(layer4_elt_sum1_3, net, net_args, engine, stream);
	// basicBlock2
	memory layer4_conv1_4 = conv2d_onednn_wo_bias(layer4_relu1_3, net, net_args, engine, stream, weightMap["layer4.1.conv1.weight"].values, t_dims, 512, 3, 3, 1, 1, 1, 1, 1, 1);
	memory layer4_bn_relu1_4 = bn_onednn(layer4_conv1_4, net, net_args, engine, stream, weightMap["layer4.1.bn1.weight"].values, weightMap["layer4.1.bn1.bias"].values, weightMap["layer4.1.bn1.running_mean"].values, weightMap["layer4.1.bn1.running_var"].values, t_dims, 1.e-5f);
	layer4_bn_relu1_4 = activation_onednn(layer4_bn_relu1_4, net, net_args, engine, stream);
	memory layer4_conv1_5 = conv2d_onednn_wo_bias(layer4_bn_relu1_4, net, net_args, engine, stream, weightMap["layer4.1.conv2.weight"].values, t_dims, 512, 3, 3, 1, 1, 1, 1, 1, 1);
	memory layer4_bn5 = bn_onednn(layer4_conv1_5, net, net_args, engine, stream, weightMap["layer4.1.bn2.weight"].values, weightMap["layer4.1.bn2.bias"].values, weightMap["layer4.1.bn2.running_mean"].values, weightMap["layer4.1.bn2.running_var"].values, t_dims, 1.e-5f);
	memory layer4_elt_sum1_5 = eltwise_onednn(layer4_relu1_3, layer4_bn5, net, net_args, engine, stream);
	memory layer4_relu1_5 = activation_onednn(layer4_elt_sum1_5, net, net_args, engine, stream);
	// layer4 
	memory global_avg_pooling = gap_onednn(layer4_relu1_5, net, net_args, engine, stream, t_dims);
	memory fc1 = fc_onednn(global_avg_pooling, net, net_args, engine, stream, weightMap["fc.weight"].values, weightMap["fc.bias"].values, t_dims, 1000);
}

int main(int argc, char **argv) {
	//std::cout << "igpu count: " << dnnl::engine::get_count(dnnl::engine::kind::gpu) << std::endl;

	// Weight load =============================================================
	// std::string file = "../model/resnet18.wts";
	// std::map<std::string, Weights> weightMap = loadWeights(file);
	// std::cout << "weight load done!" << std::endl;

	// ONEDNN =============================================================
	//[Initialize engine and stream]
	int batch_size = 1;
	int input_channel = 3;
	int input_width = 224;
	int input_height = 224;
	tensor_dims t_dims{ batch_size , input_channel, input_height, input_width };
	engine engine(engine::kind::cpu, 0);
	stream stream(engine);
	//[Create network]
	std::vector<primitive> net;
	std::vector<std::unordered_map<int, memory>> net_args;
    std::map<std::string, Weights> weightMap;
	resnet18_v2(net, net_args, engine, stream, weightMap, batch_size, input_channel, input_height, input_width, t_dims); //100th dur time : 9078 ms - > 7019 ms
	assert(net.size() == net_args.size() && "something is missing");

	// // Image load =============================================================
	// std::string img_dir = "../data";
	// std::vector<std::string> file_names;
	// if (SearchFile(img_dir.c_str(), file_names) < 0) {
	// 	std::cerr << "data search error" << std::endl;
	// 	exit(0);
	// }
	// // Image preprocess ============================================================
	// cv::Mat img(input_height, input_width, CV_8UC3);
	// cv::Mat ori_img;
	// std::vector<uint8_t> input(batch_size * input_height * input_width * input_channel);
	// std::vector<float> inputs(input.size());
	// for (int idx = 0; idx < batch_size; idx++) {
	// 	cv::Mat ori_img = cv::imread(file_names[idx]);
	// 	cv::resize(ori_img, img, img.size(), cv::INTER_LINEAR);
	// 	int offset = idx * input_height * input_width * input_channel;
	// 	memcpy(input.data() + offset, img.data, input_height * input_width * input_channel);
	// }
	// //tofile(inputs);
	// //exit(0);
	// preprocess(inputs, input, batch_size, input_channel, input_height, input_width);

	//[Execute model]
	uint64_t dur_time = 0;
	uint64_t iter_count = 1000;
    std::vector<float> user_src(batch_size * input_channel * input_width * input_height);
	//write_to_dnnl_memory(inputs.data(), net_args.at(0).find(DNNL_ARG_SRC)->second);

	for (int j = 0; j < iter_count; ++j) {
		auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

		write_to_dnnl_memory(user_src.data(), net_args.at(0).find(DNNL_ARG_SRC)->second);
		stream.wait();

		for (size_t i = 0; i < net.size(); ++i) {
			net.at(i).execute(stream, net_args.at(i));
		}

		stream.wait();
		auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - start;
		dur_time += dur;
	}

	std::vector<float> outputs(t_dims.N * t_dims.IC* t_dims.IH* t_dims.IW);
	read_from_dnnl_memory(outputs.data(), net_args.at(net.size() - 1).find(DNNL_ARG_DST)->second);
	//tofile(outputs);
	//std::cout << t_dims.N << ", "<< t_dims.IC << ", " << t_dims.IH << ", " << t_dims.IW << std::endl;
	//std::cout << "size : "<< outputs.size() << std::endl;
	//std::cout << "layer count : " << net.size() << std::endl;
	//std::cout << iter_count << " th Iteration, Total dur time :: " << dur_time << " milliseconds" << std::endl;
	//exit(0);
	//valueCheck(outputs, t_dims.N , t_dims.IC, t_dims.IH, t_dims.IW);
	// 6) ��� ���
	std::cout << "==================================================" << std::endl;
	std::cout << "===============" << " resnet18 " << "===============" << std::endl;
	std::cout << iter_count << " th Iteration, Total dur time :: " << dur_time << " milliseconds" << std::endl;
    //per iteration
    std::cout << iter_count << " th Iteration, Average dur time :: " << dur_time / iter_count << " milliseconds" << std::endl;
	// int max_index = max_element(outputs.begin(), outputs.end()) - outputs.begin();
	// std::cout << "Index : " << max_index << ", Probability : " << outputs[max_index] << ", Class Name : " << class_names[max_index] << std::endl;
	std::cout << "==================================================" << std::endl;
	std::cout << "layer count : " << net.size() << std::endl; // 2466 [ms]

	return 0;
}

memory conv2d_onednn_wo_bias(memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	std::vector<float> &weights, tensor_dims &t_dims, int OC, int KH, int KW, int SH, int SW, int TP, int BP, int LP, int RP)
{
	int OH = (t_dims.IH - KH + TP + BP) / SH + 1; // output height
	int OW = (t_dims.IW - KW + LP + RP) / SW + 1; // output width

	memory::dims conv2_src_tz = { t_dims.N, t_dims.IC, t_dims.IH, t_dims.IW };
	memory::dims conv2_weights_tz = { OC, t_dims.IC, KH, KW };
	memory::dims conv2_dst_tz = { t_dims.N, OC, OH, OW };
	memory::dims conv2_strides = { SH, SW };
	memory::dims conv2_padding1 = { TP, LP };
	memory::dims conv2_padding2 = { BP, RP };
	memory::desc conv2_dst_md = memory::desc({ conv2_dst_tz }, dt::f32, tag::any);

	// create memory for user data
	auto conv2_user_weights_memory = memory({ {conv2_weights_tz}, dt::f32, tag::oihw }, engine);
    std::vector<float> random_conv2d_weights(product(conv2_weights_tz));
	write_to_dnnl_memory(random_conv2d_weights.data(), conv2_user_weights_memory);

	// create memory descriptors for convolution data w/ no specified format
	auto conv2_src_md = memory::desc({ conv2_src_tz }, dt::f32, tag::any);
	auto conv2_weights_md = memory::desc({ conv2_weights_tz }, dt::f32, tag::any);

	// create a convolution
	auto conv2_prim_desc = convolution_forward::primitive_desc(engine,prop_kind::forward_inference, algorithm::convolution_direct,
		conv2_src_md, conv2_weights_md, conv2_dst_md, conv2_strides, conv2_padding1, conv2_padding2);

	// Activation func
	// convolution_forward::primitive_desc conv2_prim_desc = convolution_forward::primitive_desc(conv2_desc, engine);
	
	auto conv2_src_memory = INPUT;
	if (conv2_prim_desc.src_desc() != INPUT.get_desc()) {
		conv2_src_memory = memory(conv2_prim_desc.src_desc(), engine);
		net.push_back(reorder(INPUT, conv2_src_memory));
		net_args.push_back({ {DNNL_ARG_FROM, INPUT}, {DNNL_ARG_TO, conv2_src_memory} });
		engine_stream.wait();
	}
	auto conv2_weights_memory = conv2_user_weights_memory;
	if (conv2_prim_desc.weights_desc() != conv2_user_weights_memory.get_desc()) {
		conv2_weights_memory = memory(conv2_prim_desc.weights_desc(), engine);
		reorder(conv2_user_weights_memory, conv2_weights_memory).execute(engine_stream, { {DNNL_ARG_FROM, conv2_user_weights_memory},{DNNL_ARG_TO, conv2_weights_memory} });
		engine_stream.wait();
	}

	auto conv2_dst_memory = memory(conv2_prim_desc.dst_desc(), engine);

	net.push_back(convolution_forward(conv2_prim_desc));
	// Primitive arguments.
	net_args.push_back({
		{ DNNL_ARG_SRC, conv2_src_memory },
		{ DNNL_ARG_WEIGHTS, conv2_weights_memory },
		{ DNNL_ARG_DST, conv2_dst_memory }
		});

	t_dims.IC = OC;
	t_dims.IH = OH;
	t_dims.IW = OW;
	return conv2_dst_memory;
}

memory conv2d_onednn_wo_with_bias(memory &INPUT,
	std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	std::vector<float> &weights,
	tensor_dims &t_dims, int OC, int KH, int KW, int SH, int SW, int TP, int BP, int LP, int RP)
{
	int OH = (t_dims.IH - KH + TP + BP) / SH + 1; // output height
	int OW = (t_dims.IW - KW + LP + RP) / SW + 1; // output width

	memory::dims conv2_src_tz = { t_dims.N, t_dims.IC, t_dims.IH, t_dims.IW };
	memory::dims conv2_weights_tz = { OC, t_dims.IC, KH, KW };
	memory::dims conv2_dst_tz = { t_dims.N, OC, OH, OW };
	memory::dims conv2_strides = { SH, SW };
	memory::dims conv2_padding1 = { TP, LP };
	memory::dims conv2_padding2 = { BP, RP };
    memory::dims conv2_bias_tz = { OC };
	memory::desc conv2_dst_md = memory::desc({ conv2_dst_tz }, dt::f32, tag::any);
    // memory::desc conv2_bias_md = memory::desc({conv2_bias_tz}, dt::f32, tag::any);
	// create memory for user data
	auto conv2_user_weights_memory = memory({ {conv2_weights_tz}, dt::f32, tag::oihw }, engine);
    auto conv2_user_bias_memory = memory({ {conv2_bias_tz}, dt::f32, tag::x }, engine);
    std::vector<float> random_conv2d_weights(product(conv2_weights_tz));
    std::vector<float> random_conv2d_bias(product(conv2_bias_tz));
	write_to_dnnl_memory(random_conv2d_weights.data(), conv2_user_weights_memory);
    write_to_dnnl_memory(random_conv2d_bias.data(), conv2_user_bias_memory);
	// create memory descriptors for convolution data w/ no specified format
	auto conv2_src_md = memory::desc({ conv2_src_tz }, dt::f32, tag::any);
	auto conv2_weights_md = memory::desc({ conv2_weights_tz }, dt::f32, tag::any);
    auto conv2_bias_md = memory::desc({ conv2_bias_tz }, dt::f32, tag::any);

	// create a convolution
	auto conv2_prim_desc = convolution_forward::primitive_desc(engine,prop_kind::forward_inference, algorithm::convolution_direct,
		conv2_src_md, conv2_weights_md, conv2_bias_md,conv2_dst_md, conv2_strides, conv2_padding1, conv2_padding2);

	// Activation func
	// convolution_forward::primitive_desc conv2_prim_desc = convolution_forward::primitive_desc(conv2_desc, engine);
	
	auto conv2_src_memory = INPUT;
	if (conv2_prim_desc.src_desc() != INPUT.get_desc()) {
		conv2_src_memory = memory(conv2_prim_desc.src_desc(), engine);
		net.push_back(reorder(INPUT, conv2_src_memory));
		net_args.push_back({ {DNNL_ARG_FROM, INPUT}, {DNNL_ARG_TO, conv2_src_memory} });
		engine_stream.wait();
	}
	auto conv2_weights_memory = conv2_user_weights_memory;
	if (conv2_prim_desc.weights_desc() != conv2_user_weights_memory.get_desc()) {
		conv2_weights_memory = memory(conv2_prim_desc.weights_desc(), engine);
		reorder(conv2_user_weights_memory, conv2_weights_memory).execute(engine_stream, { {DNNL_ARG_FROM, conv2_user_weights_memory},{DNNL_ARG_TO, conv2_weights_memory} });
		engine_stream.wait();
	}

	auto conv2_dst_memory = memory(conv2_prim_desc.dst_desc(), engine);

	net.push_back(convolution_forward(conv2_prim_desc));
	// Primitive arguments.
	net_args.push_back({
		{ DNNL_ARG_SRC, conv2_src_memory },
		{ DNNL_ARG_WEIGHTS, conv2_weights_memory },
        { DNNL_ARG_BIAS, conv2_user_bias_memory},
		{ DNNL_ARG_DST, conv2_dst_memory }
		});

	t_dims.IC = OC;
	t_dims.IH = OH;
	t_dims.IW = OW;
	return conv2_dst_memory;
}

memory bn_onednn(memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	std::vector<float> &scale, std::vector<float> &shift, std::vector<float> &mean, std::vector<float> &var,
	tensor_dims &t_dims, float eps)
{
	std::vector<float> scale_data(t_dims.IC);
    std::vector<float> shift_data(t_dims.IC);

	// memcpy(scale_shift.data(), scale.data(), sizeof(float) * t_dims.IC);
	// memcpy(scale_shift.data() + t_dims.IC, shift.data(), sizeof(float) * t_dims.IC);

    // weightMap["bn1.weight"].values, weightMap["bn1.bias"].values, weightMap["bn1.running_mean"].values, weightMap["bn1.running_var"].values

	// auto scale_shift_mem_md = memory::desc({ 2, t_dims.IC }, dt::f32, tag::nc);
    auto scaleshift_md = memory::desc({t_dims.IC}, dt::f32, tag::x);
    auto scale_mem = memory(scaleshift_md, engine);
    auto shift_mem = memory(scaleshift_md, engine);
	// auto scale_shift_mem = memory(scale_shift_mem_md, engine);
	write_to_dnnl_memory(scale_data.data(), scale_mem);
    write_to_dnnl_memory(shift_data.data(), scale_mem);

	auto mean_mem_md = memory::desc({ 1, t_dims.IC }, dt::f32, tag::nc);
	auto mean_mem = memory(mean_mem_md, engine);
    std::vector<float> random_mean_weights(product({ 1, t_dims.IC }));
	write_to_dnnl_memory(random_mean_weights.data(), mean_mem);

	auto variance_mem_md = memory::desc({ 1, t_dims.IC }, dt::f32, tag::nc);
	auto variance_mem = memory(variance_mem_md, engine);
    std::vector<float> random_variance_weights(product({ 1, t_dims.IC }));
	write_to_dnnl_memory(random_variance_weights.data(), variance_mem);

	// Create primitive descriptor.
	batch_normalization_forward::primitive_desc bnorm_pd;

	bnorm_pd = batch_normalization_forward::primitive_desc(engine,prop_kind::forward_inference, INPUT.get_desc(),INPUT.get_desc(), 1.e-5f,
		normalization_flags::use_global_stats | normalization_flags::use_scale | normalization_flags::use_shift);
	// bnorm_pd = batch_normalization_forward::primitive_desc(bnorm_d, engine);
	
	// Create the primitive.
	auto OUTPUT = memory(bnorm_pd.dst_desc(), engine);

	net.push_back(batch_normalization_forward(bnorm_pd));
	net_args.push_back({
		{ DNNL_ARG_SRC, INPUT },
		{ DNNL_ARG_MEAN, mean_mem },
		{ DNNL_ARG_VARIANCE, variance_mem },
		{ DNNL_ARG_SCALE, scale_mem },
        { DNNL_ARG_SHIFT, shift_mem },
		{ DNNL_ARG_DST, OUTPUT }
		});
	return OUTPUT;
}

memory pooling_onednn(memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	tensor_dims &t_dims,
	int KH, int KW, int	SH, int SW, int DH, int DW, int TP, int BP, int LP, int RP,
	 int mode)
{
	const memory::dim OH = (t_dims.IH + (TP + BP) - (DH * (KH - 1) + KH)) / SH + 1;
	const memory::dim OW = (t_dims.IW + (LP + RP) - (DW * (KW - 1) + KW)) / SW + 1;

	auto pooling_dst_md = memory::desc({ t_dims.N, t_dims.IC, OH, OW }, dt::f32, tag::any);

	pooling_forward::primitive_desc pooling_pd;

	if (mode == 1) {//pooling_max
		pooling_pd = pooling_forward::primitive_desc(engine,prop_kind::forward_inference, algorithm::pooling_max,
			INPUT.get_desc(), pooling_dst_md, { SH, SW }, { KH, KW }, { DH, DW }, { TP, LP }, { BP, RP });
		// pooling_pd = pooling_v2_forward::primitive_desc(pooling_d, engine);
	}
	else {//pooling_avg
		// auto pooling_d = pooling_v2_forward::desc(prop_kind::forward_inference, algorithm::pooling_avg,
		// 	INPUT.get_desc(), pooling_dst_md, { SH, SW }, { KH, KW }, { DH, DW }, { TP, LP }, { BP, RP });
		// pooling_pd = pooling_v2_forward::primitive_desc(pooling_d, engine);
	}

	auto OUTPUT = memory(pooling_pd.dst_desc(), engine);

	net.push_back(pooling_forward(pooling_pd));
	// Primitive arguments.
	net_args.push_back({
		{ DNNL_ARG_SRC, INPUT },
		{ DNNL_ARG_DST, OUTPUT }
		});

	t_dims.IH = OH;
	t_dims.IW = OW;

	return OUTPUT;
}

memory gap_onednn(memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	tensor_dims &t_dims)
{
	const memory::dim OH = 1;
	const memory::dim OW = 1;

	auto pooling_dst_md = memory::desc({ t_dims.N, t_dims.IC, OH, OW }, dt::f32, tag::any);

	pooling_forward::primitive_desc pooling_pd;
	//pooling_avg
	pooling_pd = pooling_forward::primitive_desc(engine,prop_kind::forward_inference, algorithm::pooling_avg_exclude_padding,
		INPUT.get_desc(), pooling_dst_md, { 1, 1 }, { t_dims.IH, t_dims.IW }, { 0, 0 }, { 0, 0 }, { 0, 0 });
	// pooling_pd = pooling_v2_forward::primitive_desc(pooling_d, engine);

	auto OUTPUT = memory(pooling_pd.dst_desc(), engine);

	// Create the primitive.
	auto pooling_prim = pooling_forward(pooling_pd);

	net.push_back(pooling_prim);
	// Primitive arguments.
	net_args.push_back({
		{ DNNL_ARG_SRC, INPUT },
		{ DNNL_ARG_DST, OUTPUT }
		});

	t_dims.IH = OH;
	t_dims.IW = OW;

	return OUTPUT;
}

memory fc_onednn(memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	std::vector<float> &weights, std::vector<float> &bias,
	tensor_dims &t_dims, int OC)
{
	memory::dims fc_src_tz = { t_dims.N, t_dims.IC, 1 ,1 };
	memory::dims fc_weights_tz = { OC, t_dims.IC, 1, 1 };
	memory::dims fc_bias_tz = { OC };
	memory::dims fc_dst_tz = { t_dims.N, OC };

	auto fc_user_weights_memory = memory({ {fc_weights_tz}, dt::f32, tag::oihw }, engine);
    weights = std::vector<float>(product(fc_weights_tz));
	write_to_dnnl_memory(weights.data(), fc_user_weights_memory);

	auto fc_user_bias_memory = memory({ {fc_bias_tz}, dt::f32, tag::x }, engine);
    bias = std::vector<float>(product(fc_bias_tz));
	write_to_dnnl_memory(bias.data(), fc_user_bias_memory);

	auto fc_src_md = memory::desc({ fc_src_tz }, dt::f32, tag::any);
	auto fc_bias_md = memory::desc({ fc_bias_tz }, dt::f32, tag::any);
	auto fc_weights_md = memory::desc({ fc_weights_tz }, dt::f32, tag::any);
	auto fc_dst_md = memory::desc({ fc_dst_tz }, dt::f32, tag::any);

	// Create operation descriptor.
	auto fc_pd = inner_product_forward::primitive_desc(engine,prop_kind::forward_inference, fc_src_md, fc_weights_md, fc_bias_md, fc_dst_md);
	// auto fc_pd = inner_product_forward::primitive_desc(fc_desc, engine);

	auto fc_src_memory = INPUT;
	if (fc_pd.src_desc() != INPUT.get_desc()) {
		fc_src_memory = memory(fc_pd.src_desc(), engine);
		net.push_back(reorder(INPUT, fc_src_memory));
		net_args.push_back({ {DNNL_ARG_FROM, INPUT}, {DNNL_ARG_TO, fc_src_memory} });
		engine_stream.wait();
	}

	auto fc_weights_memory = fc_user_weights_memory;
	if (fc_pd.weights_desc() != fc_user_weights_memory.get_desc()) {
		fc_weights_memory = memory(fc_pd.weights_desc(), engine);
		reorder(fc_user_weights_memory, fc_weights_memory).execute(engine_stream, { {DNNL_ARG_FROM, fc_user_weights_memory}, {DNNL_ARG_TO, fc_weights_memory} });
		engine_stream.wait();
	}

	auto OUTPUT = memory(fc_pd.dst_desc(), engine);

	net.push_back(inner_product_forward(fc_pd));

	// Primitive arguments.
	net_args.push_back({
		{ DNNL_ARG_SRC, fc_src_memory },
		{ DNNL_ARG_WEIGHTS, fc_weights_memory },
		{ DNNL_ARG_BIAS, fc_user_bias_memory },
		{ DNNL_ARG_DST, OUTPUT }
		});

	t_dims.IC = OC;
	t_dims.IH = 1;
	t_dims.IW = 1;
	return OUTPUT;
}

memory eltwise_onednn(memory &INPUT, memory &INPUT2, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream)
{
	// Create primitive descriptor.
	auto sum_pd = sum::primitive_desc(engine,{ 1, 1 }, { INPUT.get_desc() , INPUT2.get_desc() });

	memory OUTPUT = memory(sum_pd.dst_desc(), engine);

	net.push_back(sum(sum_pd));
	// Primitive arguments.
	net_args.push_back({
		{ DNNL_ARG_MULTIPLE_SRC + 0, INPUT },
		{ DNNL_ARG_MULTIPLE_SRC + 1, INPUT2 },
		{ DNNL_ARG_DST, OUTPUT }
		});

	return OUTPUT;
}

memory activation_onednn(memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	int mode)
{
	memory OUTPUT = memory(INPUT.get_desc(), engine);

	eltwise_forward::primitive_desc eltwise_pd;

	if (mode == 0) {//relu
		eltwise_pd = eltwise_forward::primitive_desc(engine,prop_kind::forward_inference, algorithm::eltwise_relu, INPUT.get_desc(),INPUT.get_desc(), 0.f, 0.f);
		// eltwise_pd = eltwise_forward::primitive_desc(eltwise_d, engine);
	}

	net.push_back(eltwise_forward(eltwise_pd));
	net_args.push_back({
		{ DNNL_ARG_SRC, INPUT },
		{ DNNL_ARG_DST, OUTPUT }
		});

	return OUTPUT;
}

