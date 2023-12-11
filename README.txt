Author: Shaheen Ebrahimi
Advisor: Dr. Shinjiro Sueda

Program: DynamicAO
Description: Passing character joint angles into machine learning model to approximate ambient occlusion
Overview:
	The ray-tracing ambient occlusion (rtao) is used to generate high-quality samples for training using optix
	The DynamicAO (dao) uses a batched CUDA neural network evaluator to compute ao values and rasterize the result to a screen
