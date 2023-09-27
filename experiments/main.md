# Real Data
## Air36
Run GRIN model.
```bash
python run_imputation.py --model-name='grin' --dataset-name='air36' --config='imputation/grin.yaml' 
```

Run DiffGrin model, takes a long time to finish.
```bash
python run_imputation.py --model-name='diffgrin' --dataset-name='air36' --config='imputation/diffgrin_air36.yaml' 
```

Run CSDI model.
```bash
python run_imputation.py --model-name='csdi' --dataset-name='air36' --config='imputation/csdi.yaml' 
```


## Bay
Run GRIN model.
```bash
python run_imputation.py --model-name='grin' --dataset-name='bay_point' --config='imputation/grin.yaml' 
```

Run DiffGrin model.
```bash
python run_imputation.py --model-name='diffgrin' --dataset-name='bay_point' --config='imputation/diffgrin.yaml' 
```

Run Spin model.
```bash

python run_imputation.py --model-name='spin_h' --dataset-name='bay_point' --config='imputation/spin_h_bay.yaml' 
```


## Soil Moisture 
Run interpolation method.
```bash
python run_imputation.py --model-name='interpolation' --dataset-name='soil_moisture_point' 
```



Run GRIN model.
```bash
python run_imputation.py --model-name='grin' --dataset-name='soil_moisture_point' --config='imputation/grin_soil_moisture.yaml' 
``` 

Run DiffGrin model.
```bash
python run_imputation.py --model-name='diffgrin' --dataset-name='soil_moisture_point' --config='imputation/diffgrin_soil_moisture.yaml'
```

Run Spin model.
```bash
python run_imputation.py --model-name='spin_h' --dataset-name='soil_moisture_point' --config='imputation/spin_h_soil_moisture.yaml' 
```

Run CSDI model.
```bash
python run_imputation.py --model-name='csdi' --dataset-name='soil_moisture_point' --config='imputation/csdi_soil_moisture.yaml' 
```

## Soil Moisture Sparse
Run interpolation method.
```bash
python run_imputation.py --model-name='interpolation' --config='imputation/interpolation_soil_moisture.yaml'
--dataset-name='soil_moisture_sparse_point' --p-noise=0.2
```

Run Mean method.
```bash
python run_imputation.py --model-name='mean' --dataset-name='soil_moisture_sparse_point' --p-noise=0.2
```


Run GRIN model.
```bash
python run_imputation.py --model-name='grin' --dataset-name='soil_moisture_sparse_point' --config='imputation/grin_soil_moisture.yaml'
```

Run Spin model.
```bash
python run_imputation.py --model-name='spin_h' --dataset-name='soil_moisture_sparse_point' --config='imputation/spin_h_soil_moisture.yaml' 
```

Run CSDI model.
```bash
python run_imputation.py --model-name='csdi' --dataset-name='soil_moisture_sparse_point' --config='imputation/csdi_soil_moisture.yaml' 
```





# Synthetic Spatiotemporal Data
## Spatiotemporal Gaussian Process
Data is generated from spatiotemporal Gaussian Process.

Run interpolation method.
```bash
# point missing
python run_imputation.py --model-name='interpolation' --dataset-name='gp_point'

```

Run GRIN model.
```bash
# point missing
python run_imputation.py --model-name='grin' --dataset-name='gp_point' --config='imputation/grin.yaml'  

```

Run DiffGrin model.
```bash
# point missing
python run_imputation.py --model-name='diffgrin' --dataset-name='gp_point' --config='imputation/diffgrin.yaml' 


```

## Descriptive Spatiotemporal Model
Data is generated from spatiotemporal Gaussian Process.

Run interpolation method.
```bash
# point missing
python run_imputation.py --model-name='interpolation' --dataset-name='descriptive_point'


```

Run GRIN model.
```bash
# point missing
python run_imputation.py --model-name='grin' --dataset-name='descriptive_point' --config='imputation/grin.yaml' 
```

Run DiffGrin model. 
```bash
# point missing
python run_imputation.py --model-name='diffgrin' --dataset-name='descriptive_point' --config='imputation/diffgrin_descriptive.yaml'

```

## Dynamic Spatiotemporal Model

Run interpolation method and mean method.
```bash
# point missing
python run_imputation.py --model-name='interpolation' --dataset-name='dynamic_point' 

python run_imputation.py --model-name='mean' --dataset-name='dynamic_point' 
```

Run GRIN model.
```bash
# point missing
python run_imputation.py --model-name='grin' --dataset-name='dynamic_point' --config='imputation/grin.yaml' 

```

Run DiffGrin model.
```bash
# point missing
python run_imputation.py --model-name='diffgrin' --dataset-name='dynamic_point' --config='imputation/diffgrin.yaml'


``` 

Run Spin model.
```bash
# point missing
python run_imputation.py --model-name='spin_h' --dataset-name='dynamic_point' --config='imputation/spin_h.yaml' 


```
