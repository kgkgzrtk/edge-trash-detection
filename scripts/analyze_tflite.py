import tensorflow as tf

# Load the two TFLite models
model_path_1 = 'models/retrained_ssdlite_mobiledet_td_20241106072657/ssdlite_mobiledet_cpu_320x320_td_qat_20000.tflite'
model_path_2 = 'models/pretrained_model/ssdlite_mobiledet_coco_qat_postprocess/ssdlite_mobiledet_coco_qat_postprocess.tflite'

def analyze_tflite_model(model_path):
    # Load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get details about the model's input and output layers
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get the number of ops and tensors in the model
    op_details = interpreter._get_ops_details()
    num_ops = len(op_details)
    num_tensors = len(interpreter.get_tensor_details())
    
    # Collect metadata
    return {
        "input_details": input_details,
        "output_details": output_details,
        "num_ops": num_ops,
        "num_tensors": num_tensors,
        "op_types": [op['op_name'] for op in op_details]
    }

# Analyze both models
model_analysis_1 = analyze_tflite_model(model_path_1)
model_analysis_2 = analyze_tflite_model(model_path_2)

# Show the comparison results
print("Model 1 Analysis:", model_analysis_1)  # Print the analysis of the first model
print("Model 2 Analysis:", model_analysis_2)  # Print the analysis of the second model
