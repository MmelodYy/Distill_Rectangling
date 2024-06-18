<h1 align = "center">Faster,Lighter,Stronger: Image Rectangling Using Multi-Teacher Instance-Level Distillation</h1>
<p align="center">Yuan Mei*, Lichun Yang', Mengsi Wang*, Yidan GAo`, Kaijun Wu*</p>
<p align="center">* the School of Electronic and Information Engineering, Lanzhou Jiaotong University</p>
<p align="center">' the Key Lab of Opt-Electronic Technology and Intelligent Control of Ministry of Education, Lanzhou Jiaotong University</p>
<p align="center">` the School of Software Engineering, Chongqing University of Posts and Telecommunications</p>

![image](./network.png)
## Dataset (DIR-D)
We use the DIR-D dataset to train and evaluate our method. Please refer to [DeepRectangling](https://github.com/nie-lang/DeepRectangling?tab=readme-ov-file) for more details about this dataset.


## Code
#### Requirement
numpy==1.22.4

opencv_python==4.5.5.64

timm==0.9.6

torch==1.12.1+cu116

torchvision==0.13.1+cu116

## Training Teacher 2 Model

```
python train.py
```

## Training Student Model
####  Step 1:Generating an offline mesh knowledge base based on Teacher 1 and Teacher 2

```
python teacher1_gen_mesh_knowledges.py
```

```
python teacher2_gen_mesh_knowledges.py
```

####  Step 2:Training the student model

```
python train_2teachers_weight.py
```


## Testing Student Model
Our pretrained teacher 2 model and student model can be available at [Google Drive](https://drive.google.com/file/d/1LFadsV1fg-DCT9IjiKbPlIaflUdHhNVl/view?usp=sharing). Addionally, the pretrained teacher 1 model and student model can be available at [Google Drive](https://drive.google.com/drive/folders/1gEsE-7QBPcbH-kfHqYYR67C-va7vztxO?usp=sharing).
#### Caculate the rectangling performance

```
python test.py
```


## Meta
If you have any questions about this project, please feel free to drop me an email.

Yuan Mei -- 2551161628@qq.com




