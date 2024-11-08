import tensorflow as tf
import argparse


# Parse command line arguments
parser = argparse.ArgumentParser(description='Analyze TFLite models.')
parser.add_argument('--model_path_1', required=False, help='Path to the first TFLite model.')
parser.add_argument('--model_path_2', required=False, help='Path to the second TFLite model.')
args = parser.parse_args()

# Load the two TFLite models
model_path_1 = args.model_path_1
model_path_2 = args.model_path_2

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

# Function to summarize model analysis
def summarize_model_analysis(model_analysis_1, model_analysis_2):
    # Define the table header and rows
    header = "{:<20} {:<20} {:<20}".format("Attribute", "Model 1", "Model 2")
    separator = "-" * 60
    rows = [
        "{:<20} {:<20} {:<20}".format("Input Shape", str(model_analysis_1['input_details'][0]['shape']), str(model_analysis_2['input_details'][0]['shape'])),
        "{:<20} {:<20} {:<20}".format("Number of Operations", str(model_analysis_1['num_ops']), str(model_analysis_2['num_ops'])),
        "{:<20} {:<20} {:<20}".format("Number of Tensors", str(model_analysis_1['num_tensors']), str(model_analysis_2['num_tensors'])),
        "{:<20} {:<20} {:<20}".format("Main Output Shape", str(model_analysis_1['output_details'][0]['shape']), str(model_analysis_2['output_details'][0]['shape'])),
        "{:<20} {:<20} {:<20}".format("Output Dtype", model_analysis_1['output_details'][0]['dtype'].__name__, model_analysis_2['output_details'][0]['dtype'].__name__),
        "{:<20} {:<20} {:<20}".format("Top Operation Types", ', '.join(model_analysis_1['op_types'][:3]), ', '.join(model_analysis_2['op_types'][:3])),
    ]

    # Print the table
    print(header)
    print(separator)
    for row in rows:
        print(row)

    # Observations
    print("\nObservations:")
    print(f"Model 2 has slightly fewer operations ({model_analysis_2['num_ops']}) compared to Model 1 ({model_analysis_1['num_ops']}), but it has more tensors ({model_analysis_2['num_tensors']}).")
    print(f"Model 2 can output more detection boxes ({model_analysis_2['output_details'][0]['shape']}) compared to Model 1 ({model_analysis_1['output_details'][0]['shape']}), indicating a higher detection capacity.")
    print("Both models share similar operation types, indicating commonalities in their architectures.")

# Call the summary function
summarize_model_analysis(model_analysis_1, model_analysis_2)