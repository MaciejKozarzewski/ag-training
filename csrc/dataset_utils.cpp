#include <torch/extension.h>
#include <torch/library.h>

#include <alphagomoku/dataset/torch_api.h>

#include <vector>
#include <cinttypes>
#include <string>

namespace
{
	template<typename T>
	at::Tensor create_tensor(const ag::TensorSize_t &size, T dtype)
	{
		switch (size.rank)
		{
			default:
			case 0:
				return torch::zeros( { 0 }, dtype);
			case 1:
				return torch::zeros( { size.dim[0] }, dtype);
			case 2:
				return torch::zeros( { size.dim[0], size.dim[1] }, dtype);
			case 3:
				return torch::zeros( { size.dim[0], size.dim[1], size.dim[2] }, dtype);
			case 4:
				return torch::zeros( { size.dim[0], size.dim[1], size.dim[2], size.dim[3] }, dtype);
		}
	}
	ag::Sample_t tensor_to_sample_idx(const at::Tensor &t)
	{
		const int32_t *ptr = t.data_ptr<int32_t>();
		return ag::Sample_t { ptr[0], ptr[1], ptr[2], ptr[3] };
	}
}

void load_dataset_fragment(const at::Tensor &tmp, int64_t idx, const c10::string_view &path)
{
	const std::string p(path);
	ag::load_dataset_fragment(static_cast<int>(idx), p.data());
}
void unload_dataset_fragment(const at::Tensor &tmp, int64_t idx)
{
	ag::unload_dataset_fragment(static_cast<int>(idx));
}
void print_dataset_info(const at::Tensor &tmp)
{
	ag::print_dataset_info();
}
at::Tensor get_dataset_size(const at::Tensor &tmp)
{
	ag::TensorSize_t shape;
	ag::get_dataset_size(&shape, nullptr);

	assert(shape.rank == 2);
	at::Tensor result = create_tensor(shape, torch::kInt);
	ag::get_dataset_size(nullptr, result.data_ptr<int>());
	return result;
}

std::vector<at::Tensor> get_sample(const at::Tensor &sample)
{
	ag::TensorSize_t input_size;
	ag::TensorSize_t visits_size;
	ag::TensorSize_t policy_prior_size;
	ag::TensorSize_t value_target_size;
	ag::TensorSize_t minimax_value_size;
	ag::TensorSize_t minimax_score_size;
	ag::TensorSize_t moves_left_size;
	ag::TensorSize_t action_values_size;
	ag::TensorSize_t action_scores_size;

	const ag::Sample_t sample_idx = tensor_to_sample_idx(sample);

	ag::get_tensor_sizes(sample_idx, &input_size, &visits_size, &policy_prior_size, &value_target_size, &minimax_value_size, &minimax_score_size,
			&moves_left_size, &action_values_size, &action_scores_size);

	at::Tensor input = create_tensor(input_size, torch::kFloat);
	at::Tensor visits = create_tensor(visits_size, torch::kFloat);
	at::Tensor policy_prior = create_tensor(policy_prior_size, torch::kFloat);
	at::Tensor value_target = create_tensor(value_target_size, torch::kFloat);
	at::Tensor minimax_value = create_tensor(minimax_value_size, torch::kFloat);
	at::Tensor minimax_score = create_tensor(minimax_score_size, torch::kInt);
	at::Tensor moves_left = create_tensor(moves_left_size, torch::kInt);
	at::Tensor action_values = create_tensor(action_values_size, torch::kFloat);
	at::Tensor action_scores = create_tensor(action_scores_size, torch::kInt);

	ag::load_data(sample_idx, input.data_ptr<float>(), visits.data_ptr<float>(), policy_prior.data_ptr<float>(), value_target.data_ptr<float>(),
			minimax_value.data_ptr<float>(), minimax_score.data_ptr<int>(), moves_left.data_ptr<int>(), action_values.data_ptr<float>(),
			action_scores.data_ptr<int>());

	return std::vector<at::Tensor> { input, visits, policy_prior, value_target, minimax_value, minimax_score, moves_left, action_values, action_scores };
}

std::vector<at::Tensor> get_multiple_samples(const at::Tensor &samples)
{
	ag::TensorSize_t input_size;
	ag::TensorSize_t policy_target_size;
	ag::TensorSize_t value_target_size;
	ag::TensorSize_t moves_left_target_size;
	ag::TensorSize_t action_values_target_size;

	const ag::Sample_t *samples_ptr = reinterpret_cast<const ag::Sample_t*>(samples.data_ptr<int>());
	const int batch_size = samples.size(0);
	assert(samples.size(1) == 4);

	ag::get_tensor_shapes(batch_size, samples_ptr, &input_size, &policy_target_size, &value_target_size, &moves_left_target_size,
			&action_values_target_size);

	at::Tensor input = create_tensor(input_size, torch::kFloat);
	at::Tensor policy_target = create_tensor(policy_target_size, torch::kFloat);
	at::Tensor value_target = create_tensor(value_target_size, torch::kFloat);
	at::Tensor moves_left_target = create_tensor(moves_left_target_size, torch::kFloat);
	at::Tensor action_values_target = create_tensor(action_values_target_size, torch::kFloat);

	ag::load_multiple_samples(batch_size, samples_ptr, input.data_ptr<float>(), policy_target.data_ptr<float>(), value_target.data_ptr<float>(),
			moves_left_target.data_ptr<float>(), action_values_target.data_ptr<float>());

	return std::vector<at::Tensor> { input, policy_target, value_target, moves_left_target, action_values_target };
}

TORCH_LIBRARY(dataset_utils, m)
{
	m.def("load_dataset_fragment(Tensor tmp, int i, str s) -> ()");
	m.def("unload_dataset_fragment(Tensor tmp, int i) -> ()");
	m.def("print_dataset_info(Tensor tmp) -> ()");
	m.def("get_dataset_size(Tensor tmp) -> Tensor");
	m.def("get_sample(Tensor sample) -> Tensor[]");
	m.def("get_multiple_samples(Tensor sample) -> Tensor[]");
	m.impl("load_dataset_fragment", c10::DispatchKey::CPU, TORCH_FN(load_dataset_fragment));
	m.impl("unload_dataset_fragment", c10::DispatchKey::CPU, TORCH_FN(unload_dataset_fragment));
	m.impl("print_dataset_info", c10::DispatchKey::CPU, TORCH_FN(print_dataset_info));
	m.impl("get_dataset_size", c10::DispatchKey::CPU, TORCH_FN(get_dataset_size));
	m.impl("get_sample", c10::DispatchKey::CPU, TORCH_FN(get_sample));
	m.impl("get_multiple_samples", c10::DispatchKey::CPU, TORCH_FN(get_multiple_samples));
}
