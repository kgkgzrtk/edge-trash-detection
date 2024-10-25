import os
import tensorflow as tf
from object_detection import model_lib_v2

def main():
    pipeline_config_path = '../config/ssdlite_mobilenet_taco.config'
    model_dir = '../models/training'
    num_train_steps = 50000
    
    config = tf.estimator.RunConfig(model_dir=model_dir)
    
    train_and_eval_dict = {
        'pipeline_config_path': pipeline_config_path,
        'model_dir': model_dir
    }
    
    model_lib_v2.train_loop(
        pipeline_config_path=pipeline_config_path,
        model_dir=model_dir,
        train_steps=num_train_steps,
        use_tpu=False
    )

if __name__ == '__main__':
    main()
