## setup

1. Create virtual environment
	
```
$ virtualenv env
$ . env/bin/activate
```


2. Install required libraries

```
(env)$ pip install theano
(env)$ git clone git://github.com/lisa-lab/pylearn2.git
(env)$ cd pylearn2
(env)$ python setup.py develop
```


3. Modify environment variables

	- add the following lines to env/bin/activate -- they're required for python to find the random number generator in the configuration yaml file

```
PYTHONPATH=""
export PYTHONPATH
```


4. Generate the data files

	- Download the data (from kaggle); the scripts expect the data files to be in the "./data" directory relative to the .yaml and make_pylearn_data.py files
	- Run the data generator

```
(env)$ python make_pylearn_data.py
```


5. (optional) Install & configure CUDA

	- download the installer from https://developer.nvidia.com/cuda-downloads
	- add required envronmental variables http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-mac-os-x/index.html#installation
	- set up ~/.theanorc

```
[global]
device=gpu0
force_device=False
openmp=False
floatX=float32
```


6. Run the model trainer

local:
```
(env)$ pylearn2-train conv2.yaml
```

ssh:
```
(env)$ nohup caffeinate pylearn2-train conv2.yaml &
```


7. Predict new data -- see pylearn2/scripts/mlp/predict_csv.py
