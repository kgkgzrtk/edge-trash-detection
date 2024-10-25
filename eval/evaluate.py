import os
import tensorflow as tf
from object_detection import model_lib_v2

def main():
    pipeline_config_path = '../config/ssdlite_mobilenet_taco.config'
    model_dir = '../models/training'
    checkpoint_dir = model_dir
    
    config = tf.estimator.RunConfig(model_dir=model_dir)
    
    eval_config = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        steps=None,
        name='default'
    )
    
    model_lib_v2.eval_continuously(
        pipeline_config_path=pipeline_config_path,
        model_dir=model_dir,
        checkpoint_dir=checkpoint_dir,
        train_steps=None
    )

if __name__ == '__main__':
    main()
