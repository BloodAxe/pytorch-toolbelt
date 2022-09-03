@call c:\Anaconda3\Scripts\activate.bat tb
set CUDA_VISIBLE_DEVICES=
tensorboard --logdir runs --host 0.0.0.0 --port 8888 --window_title "inria_aerial"