%% Clear everything 
clearvars -global;
clear all; close all; clc;

addpath('~/ANNLib/');

[XTrain,YTrain,anglesTrain] = digitTrain4DArrayData;
[XTest,YTest,anglesTest] = digitTest4DArrayData;

[n, m, c, l] = size(XTest);

XTrainF = reshape(XTrain,[n*m,l,c]);
XTestF = reshape(XTest,[n*m,l,c]);

YTrainF = double(YTrain)';
YTestF = double(YTest)';

XTestF2 = XTestF;

XYTrainF = vertcat(XTrainF,YTrainF);
XYTestF = vertcat(XTestF,YTestF);

%%
x_off=0;
x_in=n*m+1;
t_in=1;

y_off=0;
%y_out=1;
y_out=n*m;
t_out=t_in;

ini_rate = 0.0001; 
max_epoch = 100;

regNet = Dp2BTransAEBaseNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch, 1/n);

%%
            regNet.mb_size = 256;

            regNet = Create(regNet);

%%
i=1;
        % GPU on
        gpuDevice(1);
        reset(gpuDevice(1));
    
        %regNet = regNet.Train(i, XTrainF, YTrainF);
        regNet = regNet.Train(i, XYTrainF, XTrainF);

        % GPU off
        delete(gcp('nocreate'));
        gpuDevice([]); 

%% activations
        % GPU on
        gpuDevice(1);
        reset(gpuDevice(1));
        
act1 = activations(regNet.trainedNet, XYTestF(:,:)', 'b_k_hid1');
ma = max(act1,[],'all');
mi = min(act1,[],'all');
actn = (act1 - mi)/(ma - mi);

        % GPU off
        delete(gcp('nocreate'));
        gpuDevice([]); 

%%
%i = 1 + floor(rand()*l);

%subplot(2,2,1);
%If = XTestF(:,i);
%I2 = reshape(If, [n, m]);

%image(I2 .* 255);

%subplot(2,2,2);
%Ifp = actn(:,i);
%I2p = reshape(Ifp, [n, m]);

%image(I2p .* 255);
%% test

% GPU on
gpuDevice(1);
reset(gpuDevice(1));

predictedScores = predict(regNet.trainedNet, XYTestF');
XTestF2 = predictedScores';

% GPU off
delete(gcp('nocreate'));
gpuDevice([]);

%%
i = 1 + floor(rand()*l);

subplot(2,2,1);
If = XTestF(:,i);
I2 = reshape(If, [n, m]);

image(I2 .* 255);


subplot(2,2,2);
%Ifp = XTestF2(:,i);
Ifp = actn(1:regNet.m_in-1,i);
I2p = reshape(Ifp, [n, m]);

image(I2p .* 255);


subplot(2,2,3);
I2m = I2 .* I2p;

ma = max(I2m,[],'all');
mi = min(I2m,[],'all');
I2n = (I2m - mi)/(ma - mi);

image(I2n .* 255);


subplot(2,2,4);
Ift = XTestF2(:,i);
I2t = reshape(Ift, [n, m]);

image(I2t .* 255);

XTestF2(1,i)

