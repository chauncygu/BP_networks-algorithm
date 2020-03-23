clc
clear all

load data input output

k=rand(1,2000);
[m,n]=sort(k);
input_train=input(n(1:1900),:)';
output_train=output(:,n(1:1900))';

input_test=input(n(1901:2000),:)';
output_test=output(:,n(1901:2000))';

%训练数据归一化
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train');

%构建BP神经网络训练
net=newff(inputn,outputn,[5,5]);

%网络参数设置
net.TrainParam.epochs=100;%迭代次数
net.TrainParam.rl=0.1;%学习效率
net.TrainParam.goal=0.0004;%学习目标

%开始训练神经网络
net=train(net,inputn,outputn);


%神经网络预测
[inputn_test,inputps_test]=mapminmax(input_test);

%神经网络预测结果
an=sim(net,inputn_test);

%输出结果反归一化
BPoutput=mapminmax('reverse',an,outputps);

%图形输出分析
figure(1);
plot(BPoutput,':');
hold on;
plot(output_test,'-*');


