%% tamrin olum asab_ mirmohammadsadeghi_ 98109967
%load 'C:\Users\DearUser\Downloads\AnimalDB\AnimalDB\Distractors\Bda_art142.jpg';
allImages = dir(fullfile('C:\Users\DearUser\Downloads\AnimalDB\AnimalDB\Distractors', '*.jpg'));
allanimals=  dir(fullfile('C:\Users\DearUser\Downloads\AnimalDB\AnimalDB\Targets', '*.jpg'));
allImages1=[allImages;allanimals];
%datablocks(:,:)=random('unid', length(allImages), 100, 10);
% kk=(1:1:150);
% kk=kk';
% datablocks(:,:)=crossvalind('Kfold',size(kk,1),10);
rti=randperm(150);
for i=1:1:10
    datablockstarH(:,i)=rti((i-1)*15+1:i*15);
end
%datablockstarH(:,:)=random('unid', 150, 15, 10);
rti=randperm(150);
for i=1:1:10
datablockstarC(:,i)=rti((i-1)*15+1:i*15);
end
rti=randperm(150);
for i=1:1:10
datablockstarM(:,i)=rti((i-1)*15+1:i*15);
end
rti=randperm(150);
for i=1:1:10
datablockstarF(:,i)=rti((i-1)*15+1:i*15);
end
rti=randperm(150);
for i=1:1:10
datablocksdisH(:,i)=rti((i-1)*15+1:i*15);
end
rti=randperm(150);
for i=1:1:10
datablocksdisC(:,i)=rti((i-1)*15+1:i*15);
end
rti=randperm(150);
for i=1:1:10
datablocksdisM(:,i)=rti((i-1)*15+1:i*15);
end
rti=randperm(150);
for i=1:1:10
datablocksdisF(:,i)=rti((i-1)*15+1:i*15);
end
rmdir('C:\Users\DearUser\Downloads\Image_Datasets\bckgTr', 's');
mkdir('C:\Users\DearUser\Downloads\Image_Datasets\bckgTr');

rmdir('C:\Users\DearUser\Downloads\Image_Datasets\bckgTe', 's');
mkdir('C:\Users\DearUser\Downloads\Image_Datasets\bckgTe');

rmdir('C:\Users\DearUser\Downloads\Image_Datasets\airTr', 's');
mkdir('C:\Users\DearUser\Downloads\Image_Datasets\airTr');

rmdir('C:\Users\DearUser\Downloads\Image_Datasets\airTe', 's');
mkdir('C:\Users\DearUser\Downloads\Image_Datasets\airTe');
%%
%data= VisualStimulus_data(datablocks,3);
% d=fullfile('C:\Users\DearUser\Downloads\AnimalDB\AnimalDB\Distractors',allImages(datablocks(1,1)).name)
% c=imread(d);
data1 = [];
block_number=1;
%j=[(datablocks(:,block_number))]
j=[(datablockstarC(:,block_number));(datablockstarH(:,block_number)+150);(datablockstarM(:,block_number)+300);(datablockstarF(:,block_number)+450)]
jt=[(datablocksdisC(:,block_number));(datablocksdisH(:,block_number)+150);(datablocksdisM(:,block_number)+300);(datablocksdisF(:,block_number)+450)]

% close body - far - head - medium


% tot=0;
% for p=1:1:60
%     for t=1:1:60
%         if jt(p)==jt(t)
%             if t~=p
%                 tot=tot+1;
%             end
%         end
%     end
% end
% j_tc=[(datablockstarC(:,block_number))];
% j_tm=[(datablockstarM(:,block_number))+150];
% j_tf=[(datablockstarF(:,block_number))+300];
% j_th=[(datablockstarH(:,block_number))+450];
% j_dc=[(datablocksdisC(:,block_number))];
% j_dm=[(datablocksdisM(:,block_number))+150];
% j_df=[(datablocksdisF(:,block_number))+300];
% j_dh=[(datablocksdisH(:,block_number))+450];
d=fullfile('C:\Users\DearUser\Downloads\AnimalDB\AnimalDB\Distractors',allImages(j(1),1).name)
h=imread(d);
h=rgb2gray(h)
l=fullfile('C:\Users\DearUser\Downloads\Image_Datasets',allImages(j(1),1).name)
imwrite(h,l)

j1=j;
jt1=jt;
for i=1:1:60
    d=fullfile('C:\Users\DearUser\Downloads\AnimalDB\AnimalDB\Distractors',allImages(j(i),1).name);
    c=imread(d);
    c=rgb2gray(c);
    data1(:,:,:,i)=c;
    data1or(:,:,:,i)=data1(:,:,:,i);
   data1(:,:,:,i)= (data1(:,:,:,i))./max(data1(:,:,:,i));
   l=fullfile('C:\Users\DearUser\Downloads\Image_Datasets\bckgTe',allImages(j(i),1).name)
   imwrite(c,l)
end
for i=1:1:60
    d=fullfile('C:\Users\DearUser\Downloads\AnimalDB\AnimalDB\Targets',allanimals(jt(i),1).name);
    c=imread(d);
    c=rgb2gray(c);
    data1(:,:,:,i+60)=c;
    data1or(:,:,:,i+60)=data1(:,:,:,i+60);
   data1(:,:,:,i+60)= (data1(:,:,:,i+60))./max(data1(:,:,:,i+60));
   l=fullfile('C:\Users\DearUser\Downloads\Image_Datasets\airTe',allanimals(jt(i),1).name)
   imwrite(c,l)
end
imshow(data1(:,:,:,7));
% im=data1(:,:,:,7);
% vec = randperm(numel(im));
% vec = reshape(vec, size(im));
% out = im(vec);
% imshow(out)
% %imshow((data(:,:,:,1))./max(data(:,:,:,1)));
% figure;
% imshow(h);
% im=dir(fullfile('C:\Users\DearUser\Downloads\AnimalDB\AnimalDB\Distractors',allImages(datablocks(1,1)).name));

data2 = [];
block_number=2;
j=[(datablockstarC(:,block_number));(datablockstarH(:,block_number)+150);(datablockstarM(:,block_number)+300);(datablockstarF(:,block_number)+450)]
jt=[(datablocksdisC(:,block_number));(datablocksdisH(:,block_number)+150);(datablocksdisM(:,block_number)+300);(datablocksdisF(:,block_number)+450)]
j2=j;
jt2=jt;
for i=1:1:60
    d=fullfile('C:\Users\DearUser\Downloads\AnimalDB\AnimalDB\Distractors',allImages(j(i),1).name);
    c=imread(d);
    c=rgb2gray(c);
    data2(:,:,:,i)=c;
    data2or(:,:,:,i)=data2(:,:,:,i);
   data2(:,:,:,i)= (data2(:,:,:,i))./max(data2(:,:,:,i));
   l=fullfile('C:\Users\DearUser\Downloads\Image_Datasets\bckgTr',allImages(j(i),1).name)
   imwrite(c,l)
end
for i=1:1:60
    d=fullfile('C:\Users\DearUser\Downloads\AnimalDB\AnimalDB\Targets',allanimals(jt(i),1).name);
    c=imread(d);
    c=rgb2gray(c);
    data2(:,:,:,i+60)=c;
    data2or(:,:,:,i+60)=data2(:,:,:,i+60);
   data2(:,:,:,i+60)= (data2(:,:,:,i+60))./max(data2(:,:,:,i+60));
   l=fullfile('C:\Users\DearUser\Downloads\Image_Datasets\airTr',allanimals(jt(i),1).name)
   imwrite(c,l)
end
data3 = [];
block_number=3;
j=[(datablockstarC(:,block_number));(datablockstarH(:,block_number)+150);(datablockstarM(:,block_number)+300);(datablockstarF(:,block_number)+450)]
jt=[(datablocksdisC(:,block_number));(datablocksdisH(:,block_number)+150);(datablocksdisM(:,block_number)+300);(datablocksdisF(:,block_number)+450)]
j3=j;
jt3=jt;
for i=1:1:60
    d=fullfile('C:\Users\DearUser\Downloads\AnimalDB\AnimalDB\Distractors',allImages(j(i),1).name);
    c=imread(d);
    c=rgb2gray(c);
    data3(:,:,:,i)=c;
    data3or(:,:,:,i)=data3(:,:,:,i);
   data3(:,:,:,i)= (data3(:,:,:,i))./max(data3(:,:,:,i));
   l=fullfile('C:\Users\DearUser\Downloads\Image_Datasets\bckgTr',allImages(j(i),1).name)
   imwrite(c,l)
end
for i=1:1:60
    d=fullfile('C:\Users\DearUser\Downloads\AnimalDB\AnimalDB\Targets',allanimals(jt(i),1).name);
    c=imread(d);
    c=rgb2gray(c);
    data3(:,:,:,i+60)=c;
    data3or(:,:,:,i+60)=data3(:,:,:,i+60);
   data3(:,:,:,i+60)= (data3(:,:,:,i+60))./max(data3(:,:,:,i+60));
   l=fullfile('C:\Users\DearUser\Downloads\Image_Datasets\airTr',allanimals(jt(i),1).name)
   imwrite(c,l)
end
data4 = [];
block_number=4;
j=[(datablockstarC(:,block_number));(datablockstarH(:,block_number)+150);(datablockstarM(:,block_number)+300);(datablockstarF(:,block_number)+450)]
jt=[(datablocksdisC(:,block_number));(datablocksdisH(:,block_number)+150);(datablocksdisM(:,block_number)+300);(datablocksdisF(:,block_number)+450)]
j4=j;
jt4=jt;
for i=1:1:60
    d=fullfile('C:\Users\DearUser\Downloads\AnimalDB\AnimalDB\Distractors',allImages(j(i),1).name);
    c=imread(d);
    c=rgb2gray(c);
    data4(:,:,:,i)=c;
    data4or(:,:,:,i)=data4(:,:,:,i);
   data4(:,:,:,i)= (data4(:,:,:,i))./max(data4(:,:,:,i));
   l=fullfile('C:\Users\DearUser\Downloads\Image_Datasets\bckgTr',allImages(j(i),1).name)
   imwrite(c,l)
end
for i=1:1:60
    d=fullfile('C:\Users\DearUser\Downloads\AnimalDB\AnimalDB\Targets',allanimals(jt(i),1).name);
    c=imread(d);
    c=rgb2gray(c);
    data4(:,:,:,i+60)=c;
    data4or(:,:,:,i+60)=data4(:,:,:,i+60);
   data4(:,:,:,i+60)= (data4(:,:,:,i+60))./max(data4(:,:,:,i+60));
   l=fullfile('C:\Users\DearUser\Downloads\Image_Datasets\airTr',allanimals(jt(i),1).name)
   imwrite(c,l)
end

data5 = [];
block_number=5;
j=[(datablockstarC(:,block_number));(datablockstarH(:,block_number)+150);(datablockstarM(:,block_number)+300);(datablockstarF(:,block_number)+450)]
jt=[(datablocksdisC(:,block_number));(datablocksdisH(:,block_number)+150);(datablocksdisM(:,block_number)+300);(datablocksdisF(:,block_number)+450)]
j5=j;
jt5=jt;
for i=1:1:60
    d=fullfile('C:\Users\DearUser\Downloads\AnimalDB\AnimalDB\Distractors',allImages(j(i),1).name);
    c=imread(d);
    c=rgb2gray(c);
    data5(:,:,:,i)=c;
    data5or(:,:,:,i)=data5(:,:,:,i);
   data5(:,:,:,i)= (data5(:,:,:,i))./max(data5(:,:,:,i));
   l=fullfile('C:\Users\DearUser\Downloads\Image_Datasets\bckgTr',allImages(j(i),1).name)
   imwrite(c,l)
end
for i=1:1:60
    d=fullfile('C:\Users\DearUser\Downloads\AnimalDB\AnimalDB\Targets',allanimals(jt(i),1).name);
    c=imread(d);
    c=rgb2gray(c);
    data5(:,:,:,i+60)=c;
    data5or(:,:,:,i+60)=data5(:,:,:,i+60);
   data5(:,:,:,i+60)= (data5(:,:,:,i+60))./max(data5(:,:,:,i+60));
   l=fullfile('C:\Users\DearUser\Downloads\Image_Datasets\airTr',allanimals(jt(i),1).name)
   imwrite(c,l)
end

data6 = [];
block_number=6;
j=[(datablockstarC(:,block_number));(datablockstarH(:,block_number)+150);(datablockstarM(:,block_number)+300);(datablockstarF(:,block_number)+450)]
jt=[(datablocksdisC(:,block_number));(datablocksdisH(:,block_number)+150);(datablocksdisM(:,block_number)+300);(datablocksdisF(:,block_number)+450)]
j6=j;
jt6=jt;
for i=1:1:60
    d=fullfile('C:\Users\DearUser\Downloads\AnimalDB\AnimalDB\Distractors',allImages(j(i),1).name);
    c=imread(d);
    c=rgb2gray(c);
    data6(:,:,:,i)=c;
    data6or(:,:,:,i)=data6(:,:,:,i);
   data6(:,:,:,i)= (data6(:,:,:,i))./max(data6(:,:,:,i));
   l=fullfile('C:\Users\DearUser\Downloads\Image_Datasets\bckgTr',allImages(j(i),1).name)
   imwrite(c,l)
end
for i=1:1:60
    d=fullfile('C:\Users\DearUser\Downloads\AnimalDB\AnimalDB\Targets',allanimals(jt(i),1).name);
    c=imread(d);
    c=rgb2gray(c);
    data6(:,:,:,i+60)=c;
    data6or(:,:,:,i+60)=data6(:,:,:,i+60);
   data6(:,:,:,i+60)= (data6(:,:,:,i+60))./max(data6(:,:,:,i+60));
   l=fullfile('C:\Users\DearUser\Downloads\Image_Datasets\airTr',allanimals(jt(i),1).name)
   imwrite(c,l)
end

data7 = [];
block_number=7;
j=[(datablockstarC(:,block_number));(datablockstarH(:,block_number)+150);(datablockstarM(:,block_number)+300);(datablockstarF(:,block_number)+450)]
jt=[(datablocksdisC(:,block_number));(datablocksdisH(:,block_number)+150);(datablocksdisM(:,block_number)+300);(datablocksdisF(:,block_number)+450)]
j7=j;
jt7=jt;
for i=1:1:60
    d=fullfile('C:\Users\DearUser\Downloads\AnimalDB\AnimalDB\Distractors',allImages(j(i),1).name);
    c=imread(d);
    c=rgb2gray(c);
    data7(:,:,:,i)=c;
    data7or(:,:,:,i)=data7(:,:,:,i);
   data7(:,:,:,i)= (data7(:,:,:,i))./max(data7(:,:,:,i));
   l=fullfile('C:\Users\DearUser\Downloads\Image_Datasets\bckgTr',allImages(j(i),1).name)
   imwrite(c,l)
end
for i=1:1:60
    d=fullfile('C:\Users\DearUser\Downloads\AnimalDB\AnimalDB\Targets',allanimals(jt(i),1).name);
    c=imread(d);
    c=rgb2gray(c);
    data7(:,:,:,i+60)=c;
    data7or(:,:,:,i+60)=data7(:,:,:,i+60);
   data7(:,:,:,i+60)= (data7(:,:,:,i+60))./max(data7(:,:,:,i+60));
   l=fullfile('C:\Users\DearUser\Downloads\Image_Datasets\airTr',allanimals(jt(i),1).name)
   imwrite(c,l)
end

data8 = [];
block_number=8;
j=[(datablockstarC(:,block_number));(datablockstarH(:,block_number)+150);(datablockstarM(:,block_number)+300);(datablockstarF(:,block_number)+450)]
jt=[(datablocksdisC(:,block_number));(datablocksdisH(:,block_number)+150);(datablocksdisM(:,block_number)+300);(datablocksdisF(:,block_number)+450)]
j8=j;
jt8=jt;
for i=1:1:60
    d=fullfile('C:\Users\DearUser\Downloads\AnimalDB\AnimalDB\Distractors',allImages(j(i),1).name);
    c=imread(d);
    c=rgb2gray(c);
    data8(:,:,:,i)=c;
    data8or(:,:,:,i)=data8(:,:,:,i);
   data8(:,:,:,i)= (data8(:,:,:,i))./max(data8(:,:,:,i));
   l=fullfile('C:\Users\DearUser\Downloads\Image_Datasets\bckgTr',allImages(j(i),1).name)
   imwrite(c,l)
end
for i=1:1:60
    d=fullfile('C:\Users\DearUser\Downloads\AnimalDB\AnimalDB\Targets',allanimals(jt(i),1).name);
    c=imread(d);
    c=rgb2gray(c);
    data8(:,:,:,i+60)=c;
    data8or(:,:,:,i+60)=data8(:,:,:,i+60);
   data8(:,:,:,i+60)= (data8(:,:,:,i+60))./max(data8(:,:,:,i+60));
   l=fullfile('C:\Users\DearUser\Downloads\Image_Datasets\airTr',allanimals(jt(i),1).name)
   imwrite(c,l)
end

data9 = [];
block_number=9;
j=[(datablockstarC(:,block_number));(datablockstarH(:,block_number)+150);(datablockstarM(:,block_number)+300);(datablockstarF(:,block_number)+450)]
jt=[(datablocksdisC(:,block_number));(datablocksdisH(:,block_number)+150);(datablocksdisM(:,block_number)+300);(datablocksdisF(:,block_number)+450)]
j9=j;
jt9=jt;
for i=1:1:60
    d=fullfile('C:\Users\DearUser\Downloads\AnimalDB\AnimalDB\Distractors',allImages(j(i),1).name);
    c=imread(d);
    c=rgb2gray(c);
    data9(:,:,:,i)=c;
    data9or(:,:,:,i)=data9(:,:,:,i);
   data9(:,:,:,i)= (data9(:,:,:,i))./max(data9(:,:,:,i));
   l=fullfile('C:\Users\DearUser\Downloads\Image_Datasets\bckgTr',allImages(j(i),1).name)
   imwrite(c,l)
end
for i=1:1:60
    d=fullfile('C:\Users\DearUser\Downloads\AnimalDB\AnimalDB\Targets',allanimals(jt(i),1).name);
    c=imread(d);
    c=rgb2gray(c);
    data9(:,:,:,i+60)=c;
    data9or(:,:,:,i+60)=data9(:,:,:,i+60);
   data9(:,:,:,i+60)= (data9(:,:,:,i+60))./max(data9(:,:,:,i+60));
   l=fullfile('C:\Users\DearUser\Downloads\Image_Datasets\airTr',allanimals(jt(i),1).name)
   imwrite(c,l)
end

data10 = [];
block_number=10;
j=[(datablockstarC(:,block_number));(datablockstarH(:,block_number)+150);(datablockstarM(:,block_number)+300);(datablockstarF(:,block_number)+450)]
jt=[(datablocksdisC(:,block_number));(datablocksdisH(:,block_number)+150);(datablocksdisM(:,block_number)+300);(datablocksdisF(:,block_number)+450)]
j10=j;
jt10=jt;
for i=1:1:60
    d=fullfile('C:\Users\DearUser\Downloads\AnimalDB\AnimalDB\Distractors',allImages(j(i),1).name);
    c=imread(d);
    c=rgb2gray(c);
    data10(:,:,:,i)=c;
    data10or(:,:,:,i)=data10(:,:,:,i);
   data10(:,:,:,i)= (data10(:,:,:,i))./max(data10(:,:,:,i));
   l=fullfile('C:\Users\DearUser\Downloads\Image_Datasets\bckgTr',allImages(j(i),1).name)
   imwrite(c,l)
end
for i=1:1:60
    d=fullfile('C:\Users\DearUser\Downloads\AnimalDB\AnimalDB\Targets',allanimals(jt(i),1).name);
    c=imread(d);
    c=rgb2gray(c);
    data10(:,:,:,i+60)=c;
    data10or(:,:,:,i+60)=data10(:,:,:,i+60);
   data10(:,:,:,i+60)= (data10(:,:,:,i+60))./max(data10(:,:,:,i+60));
   l=fullfile('C:\Users\DearUser\Downloads\Image_Datasets\airTr',allanimals(jt(i),1).name)
   imwrite(c,l)
end

%Data= Visual_Single_Pulse(13,"sana",23,"female",1,1,data1,data2,data3,data4,data5,data6,data7,data8,data9,data10);
%% model 
%demoRelease.m
%demonstrates how to use C2 standard model features in a pattern classification framework

addpath (genpath('C:\Users\DearUser\Downloads\osu-svm-3.0')); %put your own path to osusvm here

useSVM = 0; %if you do not have osusvm installed you can turn this
            %to 0, so that the classifier would be a NN classifier
	    %note: NN is not a great classifier for these features
	    
READPATCHESFROMFILE = 0; %use patches that were already computed
                         %(e.g., from natural images)

patchSizes = [4 8 12 16]; %other sizes might be better, maybe not
                          %all sizes are required
			  
numPatchSizes = length(patchSizes);

%specify directories for training and testing images
train_set.pos   = 'C:\Users\DearUser\Downloads\Image_Datasets\airTr';
train_set.neg   = 'C:\Users\DearUser\Downloads\Image_Datasets\bckgTr';
test_set.pos    = 'C:\Users\DearUser\Downloads\Image_Datasets\airTe';
test_set.neg    = 'C:\Users\DearUser\Downloads\Image_Datasets\bckgTe';

cI = readAllImages(train_set,test_set); %cI is a cell containing
                                        %all training and testing images

if isempty(cI{1}) | isempty(cI{2})
  error(['No training images were loaded -- did you remember to' ...
	' change the path names?']);
end
  
%below the c1 prototypes are extracted from the images/ read from file
if ~READPATCHESFROMFILE
  tic
  numPatchesPerSize = 250; %more will give better results, but will
                           %take more time to compute
                           
  cPatches = extractRandC1Patches(cI{1}, numPatchSizes, ...
      numPatchesPerSize, patchSizes); %fix: extracting from positive only 
                                      
  totaltimespectextractingPatches = toc;
else
  fprintf('reading patches');
  cPatches = load('PatchesFromNaturalImages250per4sizes','cPatches');
  cPatches = cPatches.cPatches;
end

%----Settings for Testing --------%
rot = [90 -45 0 45];
c1ScaleSS = [1:2:18];
RF_siz    = [7:2:39];
c1SpaceSS = [8:2:22];
minFS     = 7;
maxFS     = 39;
div = [4:-.05:3.2];
Div       = div;
%--- END Settings for Testing --------%

fprintf(1,'Initializing gabor filters -- full set...');
%creates the gabor filters use to extract the S1 layer
[fSiz,filters,c1OL,numSimpleFilters] = init_gabor(rot, RF_siz, Div);
fprintf(1,'done\n');

%The actual C2 features are computed below for each one of the training/testing directories
tic
for i = 1:4,
  C2res{i} = extractC2forcell(filters,fSiz,c1SpaceSS,c1ScaleSS,c1OL,cPatches,cI{i},numPatchSizes);
  toc
end
totaltimespectextractingC2 = toc;

%Simple classification code
XTrain = [C2res{1} C2res{2}]; %training examples as columns 
XTest =  [C2res{3},C2res{4}]; %the labels of the training set
ytrain = [ones(size(C2res{1},2),1);-ones(size(C2res{2},2),1)];%testing examples as columns
ytest = [ones(size(C2res{3},2),1);-ones(size(C2res{4},2),1)]; %the true labels of the test set
if useSVM
  Model = CLSosusvm(XTrain,ytrain);  %training
  [ry,rw] = CLSosusvmC(XTest,Model); %predicting new labels
else %use a Nearest Neighbor classifier
  Model = CLSnn(XTrain, ytrain); %training
  [ry,rw] = CLSnnC(XTest,Model); %predicting new labels
end 
successrateC = sum(ytest(1:15)==ry(1:15))+sum(ytest(60:75)==ry(60:75)); %a simple classification score
successrateF = sum(ytest(15:30)==ry(15:30))+sum(ytest(75:90)==ry(75:90));
successrateH = sum(ytest(30:45)==ry(30:45))+sum(ytest(90:105)==ry(90:105));
successrateM = sum(ytest(45:60)==ry(45:60))+sum(ytest(105:120)==ry(105:120));

accuracymodel(:)=[successrateC/30;successrateF/30;successrateH/30;successrateM/30];


%%
%getting the test
%% sub 1
numsub=1;
[durrespC(numsub),durrespH(numsub),durrespM(numsub),durrespF(numsub),successnC(numsub),successnF(numsub),successnH(numsub),successnM(numsub),successpC(numsub),successpF(numsub),successpH(numsub),successpM(numsub)]=test(data1or)
%% sub 2
numsub=2;
[durrespC(numsub),durrespH(numsub),durrespM(numsub),durrespF(numsub),successnC(numsub),successnF(numsub),successnH(numsub),successnM(numsub),successpC(numsub),successpF(numsub),successpH(numsub),successpM(numsub)]=test(data1or)
%% sub 3
numsub=3;
[durrespC(numsub),durrespH(numsub),durrespM(numsub),durrespF(numsub),successnC(numsub),successnF(numsub),successnH(numsub),successnM(numsub),successpC(numsub),successpF(numsub),successpH(numsub),successpM(numsub)]=test(data1or)



%% plot accuracies
for n=1:1:numsub
    accuracyC(n)=(successnC(n)+successpC(n))/30;
    accuracyH(n)=(successnH(n)+successpH(n))/30;
    accuracyM(n)=(successnM(n)+successpM(n))/30;
    accuracyF(n)=(successnF(n)+successpF(n))/30;
    accuracy(n,:)=[accuracyC(n);accuracyH(n);accuracyM(n);accuracyF(n)];
end

col={'b-*','r-o','g-+','c-s'};
TW=1:1:4
for mth=1:3
    plot(TW,accuracy(mth,:),col{mth},'LineWidth',1);
    hold on;
end

plot(TW,accuracymodel(:),col{4},'LineWidth',1);
hold on;
xlabel('group of image(close,head,medium,far)');
ylabel('Accuracy (%)');
grid;
xlim([0.75 4.25]);
ylim([0 1.25]);
set(gca,'xtick',1:5,'xticklabel',1:5);
title('\bf accuracy');
h=legend({'subject1','subject2','subject3','model'});
set(h,'Location','SouthEast');
%% plot response time

for n=1:1:numsub
    durresp(n,:)=[durrespC(n);durrespH(n);durrespM(n);durrespF(n)];
end
col={'b-*','r-o','g-+'};
TW=1:1:4;
for mth=1:3
    plot(TW,durresp(mth,:),col{mth},'LineWidth',1);
    hold on;
end
xlabel('group of image(close,head,medium,far)');
ylabel('duration of response');
grid;
xlim([0.75 4.25]);
ylim([0 2]);
set(gca,'xtick',1:5,'xticklabel',1:5);
title('\bf response time');
h=legend({'subject1','subject2','subject3'});
set(h,'Location','SouthEast');

%% Displaying images
function [durrespCt,durrespHt,durrespMt,durrespFt,successnC,successnF,successnH,successnM,successpC,successpF,successpH,successpM]=test(data1or)
Screen('Preference', 'SkipSyncTests', 1);
%Open the screen
[wPtr,rect]=Screen('Openwindow',max(Screen('Screens')));
k=randperm(120);
% k=(1:1:120);
%Create texture
successpC=0;
successpH=0;
successpM=0;  
successpF=0;

successnC=0;
successnH=0;
successnM=0;
successnF=0;
s=0;
durrespC=[];
durrespH=[];
durrespM=[];
durrespF=[];
numimC=1;
numimH=1;
numimM=1;
numimF=1;
for i=1:1:120
    k(i)
faceTexture=Screen('MakeTexture',wPtr,data1or(:,:,:,k(i)));

%Draw it 
Screen('DrawTexture',wPtr,faceTexture);
i1=GetSecs();
Screen('Flip',wPtr); 
i2=GetSecs();
%Wait for keypress and clear
i3=i2-i1;
WaitSecs(0.02-i3);
%KbWait(); 
Screen('Flip',wPtr); 
WaitSecs(0.03);
im=data1or(:,:,:,k(i));
vec = randperm(numel(im));
vec = reshape(vec, size(im));
out = im(vec);
imshow(out)
maskTexture=Screen('MakeTexture',wPtr,out);

%Draw it
Screen('DrawTexture',wPtr,maskTexture);
Screen('Flip',wPtr); 
WaitSecs(0.1); 
startresp=GetSecs();
[~, startrt] = Screen('Flip',wPtr);
    offTime_stim = startrt;
%WaitSecs(20); 
DataFinaal_Responsesum=0;
keyIsDown = 0;
    flag = 0;
    DataResponse_Duration=26;
    while ~flag && (GetSecs - offTime_stim)<=(DataResponse_Duration)
        [keyIsDown, secs, keyCode] = KbCheck(); %#ok<*ASGLU>
        if keyIsDown
            offTime_response=GetSecs();
            a = KbName(keyCode);
            hh = strcmp(a ,'n');
            gg = strcmp(a ,'y');
            
            if (strcmp(a(1) ,'n') || hh(1))
                DataFinaal_Response = 0;
                flag = 1;
                % DataResponseTime(tt) = offTime_response -  offTime_stim;
                
            elseif  (strcmp(a(1) ,'y') || gg(1))
                DataFinaal_Response = 1;
                flag = 1;
                % DataResponseTime(tt) = offTime_response -  offTime_stim;
                
            else
                DataFinaal_Response =0;
            end
        end
    end
    if gg(1)
        s=s+1;
    end
    if ((k(i)<=15)|| (((60<k(i))&&(k(i)<=75))))
        durrespC(numimC)= offTime_response-startresp;
            numimC=numimC+1;
    end
    if (((15<k(i))&&(k(i)<=30))|| ((75<k(i))&&(k(i)<=90)))
        durrespF(numimF)= offTime_response-startresp;
            numimF=numimF+1;
    end
     if (((30<k(i))&&(k(i)<=45))|| ((90<k(i))&&(k(i)<=105)))
        durrespH(numimH)= offTime_response-startresp;
            numimH=numimH+1;
     end
     if (((45<k(i))&&(k(i)<=60))|| ((105<k(i))&&(k(i)<=120)))
        durrespM(numimM)= offTime_response-startresp;
            numimM=numimM+1;
    end
    if ((k(i)<=15)&& (~gg(1)))
%         if  ~gg(1)
            successnC=successnC+1;
            
%         end
    end
    if (((15<k(i))&&(k(i)<=30))&& (~gg(1)))
%         if  ~gg(1)
            successnF=successnF+1;
%         end
    end
   if (((30<k(i))&&(k(i)<=45))&& (~gg(1)))
%      if  ~gg(1)
         successnH=successnH+1;
%      end
   end
   if (((45<k(i))&&(k(i)<=60))&& (~gg(1))) 
%      if  ~gg(1)
         successnM=successnM+1;
%      end 
   end
   if (((60<k(i))&&(k(i)<=75))&& (gg(1)))
%      if  gg(1)
         successpC=successpC+1;
%      end
   end
   if (((75<k(i))&&(k(i)<=90))&& (gg(1))) 
%      if  gg(1)
         successpF=successpF+1;
%      end
   end
   if (((90<k(i))&&(k(i)<=105))&& (gg(1)))
%        if  gg(1)
           successpH=successpH+1;
%        end
   end
   if (((105<k(i))&&(k(i)<=120))&& (gg(1)))
%        if  gg(1)
           successpM=successpM+1;
%        end
   end
   
     
end
durrespCt=sum(durrespC)/30;
durrespHt=sum(durrespH)/30;
durrespMt=sum(durrespM)/30;
durrespFt=sum(durrespF)/30;
%     DataFinaal_Responsesum=DataFinaal_Responsesum+DataFinaal_Response;
%     faceTexture=Screen('MakeTexture',wPtr,data1(:,:,:,5));
% %Draw it 
% Screen('DrawTexture',wPtr,faceTexture);
% 
% Screen('Flip',wPtr); 
% 
% %Wait for keypress and clear
% WaitSecs(2);
% %KbWait(); 
% Screen('Flip',wPtr); 
% WaitSecs(3);
% maskTexture=Screen('MakeTexture',wPtr,data1(:,:,:,5));
% 
% %Draw it
% Screen('DrawTexture',wPtr,maskTexture);
% Screen('Flip',wPtr); 
% WaitSecs(2); 
% 
% [~, startrt] = Screen('Flip',wPtr);
%     offTime_stim = startrt;
% %WaitSecs(20); 
% keyIsDown = 0;
%     flag = 0;
%     DataResponse_Duration=26;
%     while ~flag && (GetSecs - offTime_stim)<=(DataResponse_Duration)
%         [keyIsDown, secs, keyCode] = KbCheck(); %#ok<*ASGLU>
%         if keyIsDown
%             offTime_response=GetSecs();
%             a = KbName(keyCode);
%             hh = strcmp(a ,'d');
%             gg = strcmp(a ,'a');
%             
%             if (strcmp(a(1) ,'d') || hh(1))
%                 DataFinaal_Response = 0;
%                 flag = 1;
%                 % DataResponseTime(tt) = offTime_response -  offTime_stim;
%                 
%             elseif  (strcmp(a(1) ,'a') || gg(1))
%                 DataFinaal_Response = 1;
%                 flag = 1;
%                 % DataResponseTime(tt) = offTime_response -  offTime_stim;
%                 
%             else
%                 DataFinaal_Response =0;
%             end
%         end
%     end
%     DataFinaal_Responsesum=DataFinaal_Responsesum+DataFinaal_Response;
% %clear Screen;
% %KbWait(); 
% %screens = Screen('Screens');
% % screen_struct.cur_window = max(screens);
% % gray = GrayIndex(screen_struct.cur_window, 0.5);
% % [w, screen_struct.wRect] = Screen('OpenWindow',screen_struct.cur_window, gray);
% % Screen('Flip', w);
KbWait();
% 
 clear Screen;
end
%%
function data= VisualStimulus_data(datablocks,block_number)

end
%%


%%
%% c1
function [c1,s1] = C1(stim, filters, fSiz, c1SpaceSS, c1ScaleSS, c1OL,INCLUDEBORDERS)
%function [c1,s1] = C1(stim, filters, fSiz, c1SpaceSS, c1ScaleSS, c1OL,INCLUDEBORDERS)
%
%  A matlab implementation of the C1 code originally by Max Riesenhuber
%  and Thomas Serre.
%  Adapted by Stanley Bileschi
%
%  Returns the C1 and the S1 units' activation given the  
%  input image, stim.
%  filters, fSiz, c1ScaleSS, c1ScaleSS, c1OL, INCLUDEBORDERS are the
%  parameters of the C1 system
%
%  stim   - the input image must be grayscale (single channel) and 
%   type ''double''
%
%%% For S1 unit computation %%%
%
% filters -  Matrix of Gabor filters of size max_fSiz x num_filters,
% where max_fSiz is the length of the largest filter and num_filters the
% total number of filters. Column j of filters matrix contains a n_jxn_j
% filter (reshaped as a column vector and padded with zeros).
%
% fSiz - Vector of size num_filters containing the various filter
% sizes. fSiz(j) = n_j if filters j is n_j x n_j (see variable filters
% above).
%
%%% For C1 unit computation %%%
%
% c1ScaleSS  - Vector defining the scale bands, i.e. a group of filter
% sizes over which a local max is taken to get the C1 unit responses,
% e.g. c1ScaleSS = [1 k num_filters+1] means 2 scale bands, the first
% one contains filters(:,1:k-1) and the second one contains
% filters(:,k:num_filters). If N pooling bands, c1ScaleSS should be of
% length N+1.
%
% c1SpaceSS - Vector defining the spatial pooling range for each scale
% band, i.e. c1SpaceSS(i) = m_i means that each C1 unit response in band
% i is obtained by taking a max over a local neighborhood of m_ixm_i S1
% units. If N bands then c1SpaceSS should be of size N.
%
% c1OL - Scalar value defining the overlap between C1 units. In scale
% band i, the C1 unit responses are computed every c1Space(i)/c1OL.
%
% INCLUDEBORDERS - the type of treatment for the image borders.

USECONV2 = 0; %should be faster if 1

USE_NORMXCORR_INSTEAD = 0;
if(nargin < 7)
  INCLUDEBORDERS = 0;
end
numScaleBands=length(c1ScaleSS)-1;  % convention: last element in c1ScaleSS is max index + 1 
numScales=c1ScaleSS(end)-1; 
%   last index in scaleSS contains scale index where next band would start, i.e., 1 after highest scale!!
numSimpleFilters=floor(length(fSiz)/numScales);
for iBand = 1:numScaleBands
  ScalesInThisBand{iBand} = c1ScaleSS(iBand):(c1ScaleSS(iBand+1) -1);
end  

% Rebuild all filters (of all sizes)
%%%%%%%%
nFilts = length(fSiz);
for i = 1:nFilts
  sqfilter{i} = reshape(filters(1:(fSiz(i)^2),i),fSiz(i),fSiz(i));
  if USECONV2
    sqfilter{i} = sqfilter{i}(end:-1:1,end:-1:1); %flip in order to use conv2 instead of imfilter (%bug_fix 6/28/2007);
  end    
end

% Calculate all filter responses (s1)
%%%%%%%%
sqim = stim.^2;
iUFilterIndex = 0;
% precalculate the normalizations for the usable filter sizes
uFiltSizes = unique(fSiz);
for i = 1:length(uFiltSizes)
  s1Norm{uFiltSizes(i)} = (sumfilter(sqim,(uFiltSizes(i)-1)/2)).^0.5;
  %avoid divide by zero
  s1Norm{uFiltSizes(i)} = s1Norm{uFiltSizes(i)} + ~s1Norm{uFiltSizes(i)};
end

for iBand = 1:numScaleBands
   for iScale = 1:length(ScalesInThisBand{iBand})
     for iFilt = 1:numSimpleFilters
       iUFilterIndex = iUFilterIndex+1;
       if ~USECONV2
	 s1{iBand}{iScale}{iFilt} = abs(imfilter(stim,sqfilter{iUFilterIndex},'symmetric','same','corr'));

	 if(~INCLUDEBORDERS)
	   s1{iBand}{iScale}{iFilt} = removeborders(s1{iBand}{iScale}{iFilt},fSiz(iUFilterIndex));
	 end
	 s1{iBand}{iScale}{iFilt} = im2double(s1{iBand}{iScale}{iFilt}) ./ s1Norm{fSiz(iUFilterIndex)};
       else %not 100% compatible but 20% faster at least
	 s1{iBand}{iScale}{iFilt} = abs(conv2(stim,sqfilter{iUFilterIndex},'same'));
	 if(~INCLUDEBORDERS)
	   s1{iBand}{iScale}{iFilt} = removeborders(s1{iBand}{iScale}{iFilt},fSiz(iUFilterIndex));
	 end
	 s1{iBand}{iScale}{iFilt} = im2double(s1{iBand}{iScale}{iFilt}) ./ s1Norm{fSiz(iUFilterIndex)};
       end
     end
   end
end


% Calculate local pooling (c1)
%%%%%%%%

%   (1) pool over scales within band
for iBand = 1:numScaleBands
  for iFilt = 1:numSimpleFilters
    c1{iBand}(:,:,iFilt) = zeros(size(s1{iBand}{1}{iFilt}));
    for iScale = 1:length(ScalesInThisBand{iBand});
      c1{iBand}(:,:,iFilt) = max(c1{iBand}(:,:,iFilt),s1{iBand}{iScale}{iFilt});
    end
  end
end

%   (2) pool over local neighborhood
for iBand = 1:numScaleBands
  poolRange = (c1SpaceSS(iBand));
  for iFilt = 1:numSimpleFilters
    c1{iBand}(:,:,iFilt) = maxfilter(c1{iBand}(:,:,iFilt),[0 0 poolRange-1 poolRange-1]);
  end
end


%   (3) subsample
for iBand = 1:numScaleBands
  sSS=ceil(c1SpaceSS(iBand)/c1OL);
  clear T;
  for iFilt = 1:numSimpleFilters
    T(:,:,iFilt) = c1{iBand}(1:sSS:end,1:sSS:end,iFilt);
  end
  c1{iBand} = T;
end


function sout = removeborders(sin,siz)
sin = unpadimage(sin, [(siz+1)/2,(siz+1)/2,(siz-1)/2,(siz-1)/2]);
sin = padarray(sin, [(siz+1)/2,(siz+1)/2],0,'pre');
sout = padarray(sin, [(siz-1)/2,(siz-1)/2],0,'post');
       

end
end

%% C2
function [c2,s2,c1,s1] = C2(stim,filters,fSiz,c1SpaceSS,c1ScaleSS,c1OL,s2Target,c1)
%function [c2,s2,c1,s1] = C2(stim,filters,fSiz,c1SpaceSS,c1ScaleSS,c1OL,s2Target,c1)
%
% given an image extracts layers s1 c1 s2 and finally c2
% for inputs stim, filters, fSiz, c1SpaceSS,c1ScaleeSS, and c1OL
% see the documentation for C1 (C1.m)
%
% briefly, 
% stim is the input image. 
% filters fSiz, c1SpaceSS, c1ScaleSS, c1OL are the parameters of
% the c1 process
%
% s2Target are the prototype (patches) to be used in the extraction
% of s2.  Each patch of size [n,n,d] is stored as a column in s2Target,
% which has itself a size of [n*n*d, n_patches];
%
% if available, a precomputed c1 layer can be used to save computation
% time.  The proper format is the output of C1.m
%
% See also C1


if nargin<8
  [c1,s1] = C1(stim,filters,fSiz,c1SpaceSS,c1ScaleSS,c1OL);
end

nbands = length(c1);
c1BandImage = c1;
nfilts = size(c1{1},3);
n_rbf_centers = size(s2Target,2);
L = size(s2Target,1) / nfilts;
PatchSize = [L^.5,L^.5,nfilts];

s2 = cell(n_rbf_centers,1);

%Build s2:
%  for all prototypes in s2Target (RBF centers)
%   for all bands
%    calculate the image response
for iCenter = 1:n_rbf_centers
  Patch = reshape(s2Target(:,iCenter),PatchSize);
  s2{iCenter} = cell(nbands,1);
  for iBand = 1:nbands
     s2{iCenter}{iBand} = WindowedPatchDistance(c1BandImage{iBand},Patch);  
  end
end

%Build c2:
% calculate minimum distance (maximum stimulation) across position and scales
c2 = inf(n_rbf_centers,1);
for iCenter = 1:n_rbf_centers
  for iBand = 1:nbands
     c2(iCenter) = min(c2(iCenter),min(min(s2{iCenter}{iBand})));
  end
end
end

%% CLSosusvm
function Model = CLSosusvm(Xtrain,Ytrain,sPARAMS);
%function Model = CLSosusvm(Xtrain,Ytrain,sPARAMS);
%
%Builds an SVM classifier
%This is only a wrapper function for osu svm
%It requires that osu svm (http://www.ece.osu.edu/~maj/osu_svm/) is installed and included in the path
%X contains the data-points as COLUMNS, i.e., X is nfeatures \times nexamples
%y is a column vector of all the labels. y is nexamples \times 1
%sPARAMS is a structure of parameters:
%sPARAMS.KERNEL specifies the kernel type
%sPARAMS.C specifies the regularization constant
%sPARAMS.GAMMA, sPARAMS.DEGREE are parameters of the kernel function
%Model contains the parameters of the SVM model as returned by osu svm

Ytrain = Ytrain';
if nargin<3
  SETPARAMS = 1;
elseif isempty(sPARAMS)
  SETPARAMS = 1;
else
  SETPARAMS = 0;
end

if SETPARAMS
  sPARAMS.KERNEL = 0;
  sPARAMS.C = 1;
end

switch sPARAMS.KERNEL,
  case 0,
    [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = ...
	LinearSVC(Xtrain, Ytrain, sPARAMS.C);
  case 1,
    [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = ...
	PolySVC(Xtrain, Ytrain, sPARAMS.DEGREE, sPARAMS.C, 1,0);
  case 2,
    [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = ...
	PolySVC(Xtrain, Ytrain, sPARAMS.DEGREE, sPARAMS.C, 1,sPARAMS.COEF);
  case 3,
    [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = ...
	RbfSVC(Xtrain, Ytrain, sPARAMS.GAMMA, sPARAMS.C);    
end

Model.AlphaY = AlphaY;
Model.SVs = SVs;
Model.Bias = Bias;
Model.Parameters = Parameters;
Model.nSV = nSV;
Model.nLabel = nLabel;
Model.sPARAMS = sPARAMS;

end

%% CLSnnC
function [labels,weights,firstindeces] = CLSnnC(X,Model);
%function [labels,weights,firstindeces] = CLSnnC(X,Model);
%
%X contains the data-points to be classified as COLUMNS, i.e., it is nfeatures \times nexamples
%Model is the model returned by CLSnn
%labels are the predicted labels
%
%
%Inputs:
%Model.k
%Model.trainX 
%Model.trainy 
%
%See also CLSnn
if isfield(Model,'deg')
  deg = Model.deg;
else 
  deg = 2;
end

len1 = size(X,2);
len2 = size(Model.trainX,2);
if deg==2
  X2 = sum((X).^2,1);
  Z2 = sum((Model.trainX).^2,1);
  distance = (repmat(Z2,len1,1)+repmat(X2',1,len2)-2*X'*Model.trainX)';
else
  distance = zeros(len2,len1);
  for i = 1:len1,
    for j = 1:len2,
      distance(j,i) = sum(abs(X(:,i)-Model.trainX(:,j)).^deg);
    end
  end
end

[sorted,index] = sort(distance);
yy = Model.trainy(index);
if Model.k>1
  weights = mean(yy(1:Model.k,:),1)';
  disp('kNN weights::just sign no voting');
  labels = sign(weights);
else
  labels = yy(1,:)';
  weights = 1./(sorted(1,:)'+eps);
end

if nargout>2
  numindeces = min(size(sorted,1),Model.numindeces);
  firstindeces = yy(1:numindeces,:)';
end

end

%% CLSnn
function [Model,looerrors] = CLSnn(X,y,sPARAMS);
%function [Model,looerrors] = CLSnn(X,y,sPARAMS);
%
%Builds a NN classifier
%X contains the data-points as COLUMNS, i.e., X is nfeatures \times nexamples
%y is a column vector of all the labels. y is nexamples \times 1
%sPARAMS is a structure of parameters:
%sPARAMS.k is the k for knn
%sPARAMS.deg determines the p-norm to be used as distance
%Model contains the parameters of the nn classifier 

if nargin<3
  sPARAMS.k = 1;
end

if ~isfield(sPARAMS,'deg')
  sPARAMS.deg = 2;
end

Model.k = sPARAMS.k;
Model.deg = sPARAMS.deg;
Model.trainX = X;
Model.trainy = y;

if isfield(sPARAMS,'numindeces')
  Model.numindeces = sPARAMS.numindeces;
else
  Model.numindeces = inf;
end

if nargout>1
  deg = Model.deg;
  len1 = size(X,2);
  len2 = size(Model.trainX,2);
  if deg==2
    X2 = sum((X).^2,1);
    Z2 = sum((Model.trainX).^2,1);
    distance = (repmat(Z2,len1,1)+repmat(X2',1,len2)-2*X'*Model.trainX)';
  else
    distance = zeros(len2,len1);
    for i = 1:len1,
      for j = 1:len2,
	distance(j,i) = sum(abs(X(:,i)-Model.trainX(:,j)).^deg);
      end
    end
  end
  
  maxdistance = max(distance(:));
  distance = distance + eye(len1)*maxdistance;
  
  [sorted,index] = sort(distance);
  yy = Model.trainy(index);
  if Model.k>1
    weights = mean(yy(1:Model.k,:),1)';
    disp('kNN weights::just sign no voting');
    labels = sign(weights);
  else
    labels = yy(1,:)';
    weights = 1./(sorted(1,:)'+eps);
  end
  looerrors = mean(labels==y);
end
end

%% extractC2forcell
function mC2 = extractC2forcell(filters,fSiz,c1SpaceSS,c1ScaleSS,c1OL,cPatches,cImages,numPatchSizes);
%function mC2 = extractC2forcell(filters,fSiz,c1SpaceSS,c1ScaleSS,c1OL,cPatches,cImages,numPatchSizes);
%
%this function is a wrapper of C2. For each image in the cell cImages, 
%it extracts all the values of the C2 layer 
%for all the prototypes in the cell cPatches.
%The result mC2 is a matrix of size total_number_of_patches \times number_of_images where
%total_number_of_patches is the sum over i = 1:numPatchSizes of length(cPatches{i})
%and number_of_images is length(cImages)
%The C1 parameters used are given as the variables filters,fSiz,c1SpaceSS,c1ScaleSS,c1OL
%for more detail regarding these parameters see the help entry for C1
%
%See also C1

%a bug was fixed on Jul 01 2005

numPatchSizes = min(numPatchSizes,length(cPatches));
%all the patches are being flipped. This is becuase in matlab conv2 is much faster than filter2
for i = 1:numPatchSizes,
  [siz,numpatch] = size(cPatches{i});
  siz = sqrt(siz/4);
  for j = 1:numpatch,
    tmp = reshape(cPatches{i}(:,j),[siz,siz,4]);
    tmp = tmp(end:-1:1,end:-1:1,:);
    cPatches{i}(:,j) = tmp(:);
  end
end

mC2 = [];

for i = 1:length(cImages), %for every input image
  fprintf(1,'%d:',i);
  stim = cImages{i};
  img_siz = size(stim);
  c1  = [];
  iC2 = []; %bug fix
  for j = 1:numPatchSizes, %for every unique patch size
    fprintf(1,'.');
    if isempty(c1),  %compute C2
      [tmpC2,tmp,c1] = C2(stim,filters,fSiz,c1SpaceSS,c1ScaleSS,c1OL,cPatches{j});
    else
      [tmpC2] = C2(stim,filters,fSiz,c1SpaceSS,c1ScaleSS,c1OL,cPatches{j},c1);
    end
    iC2 = [iC2;tmpC2];
  end
  mC2 = [mC2, iC2];
end
fprintf('\n');
end

%% CLSosusvmC
function [Labels, DecisionValue]= CLSosusvmC(Samples, Model);
%function [Labels, DecisionValue]= CLSosusvmC(Samples, Model);
%
%wrapper function for osu svm classification
%Samples contains the data-points to be classified as COLUMNS, i.e., it is nfeatures \times nexamples
%Model is the model returned by CLSosusvm
%Labels are the predicted labels
%DecisionValue are the values assigned by the Model to the points (Labels = sign(DecisionValue))

[Labels, DecisionValue]= SVMClass(Samples, Model.AlphaY, ...
                                  Model.SVs, Model.Bias, ...
				  Model.Parameters, Model.nSV, Model.nLabel);
Labels = Labels';
DecisionValue = DecisionValue';
end

%% extractRandC1Patches
function cPatches = extractRandC1Patches(cItrainingOnly, numPatchSizes, numPatchesPerSize, patchSizes);
%extracts random prototypes as part of the training of the C2 classification 
%system. 
%Note: we extract only from BAND 2. Extracting from all bands might help
%cPatches the returned prototypes
%cItrainingOnly the training images
%numPatchesPerSize is the number of sizes in which the prototypes come
%numPatchesPerSize is the number of prototypes extracted for each size
%patchSizes is the vector of the patche sizes

if nargin<2
  numPatchSizes = 4;
  numPatchesPerSize = 250;
  patchSizes = 4:4:16;
end

nImages = length(cItrainingOnly);

%----Settings for Training the random patches--------%
rot = [90 -45 0 45];
c1ScaleSS = [1 3];
RF_siz    = [11 13];
c1SpaceSS = [10];
minFS     = 11;
maxFS     = 13;
div = [4:-.05:3.2];
Div       = div(3:4);
%--- END Settings for Training the random patches--------%

fprintf(1,'Initializing gabor filters -- partial set...');
[fSiz,filters,c1OL,numSimpleFilters] = init_gabor(rot, RF_siz, Div);
fprintf(1,'done\n');

cPatches = cell(numPatchSizes,1);
bsize = [0 0];

pind = zeros(numPatchSizes,1);
for j = 1:numPatchSizes
  cPatches{j} = zeros(patchSizes(j)^2*4,numPatchesPerSize);
end

for i = 1:numPatchesPerSize,
  ii = floor(rand*nImages) + 1;
  fprintf(1,'.');
  stim = cItrainingOnly{ii};
  img_siz = size(stim);
  
  [c1source,s1source] = C1(stim, filters, fSiz, c1SpaceSS, ...
      c1ScaleSS, c1OL);
  b = c1source{1}; %new C1 interface;
  bsize(1) = size(b,1);
  bsize(2) = size(b,2);
  for j = 1:numPatchSizes,
    xy = floor(rand(1,2).*(bsize-patchSizes(j)))+1;
    tmp = b(xy(1):xy(1)+patchSizes(j)-1,xy(2):xy(2)+patchSizes(j)-1,:);
    pind(j) = pind(j) + 1;
    cPatches{j}(:,pind(j)) = tmp(:); 
  end
end
fprintf('\n');
end

%% maxfilter
function I = maxfilter(I,radius)
%function I = maxfilter(I,radius)
%
%Performs morphological dilation on a multilayer image.
%
%I is the input image
%radius is the additional radius of the window, i.e., 5 means 11 x 11
%if a four value vector is specified for radius, then any rectangular support may be used for max.
%in the order left top right bottom.
switch length(radius)
case 1,
  I = padimage(I,radius);
  [n,m,thirdd] = size(I);
  B = I;
  for i = radius+1:m-radius,
    B(:,i,:) = max(I(:,i-radius:i+radius,:),[],2);
  end
  for i = radius+1:n-radius,
    I(i,:,:) = max(B(i-radius:i+radius,:,:),[],1);
  end
  I = unpadimage(I,radius);
case 4,
  [n,m,thirdd] = size(I);
  B = I;
  for i=1:radius(1)
    B(:,i,:) = max(I(:,max(1,i-radius(1)):min(end,i+radius(3)),:),[],2);
  end
  for i = radius(1)+1:m-radius(3),
    B(:,i,:) = max(I(:,i-radius(1):i+radius(3),:),[],2);
  end
  for i=m-radius(3)+1:m
    B(:,i,:) = max(I(:,i-radius(1):min(end,i+radius(3)),:),[],2);
  end
  for i = 1:radius(2),
    I(i,:,:) = max(B(max(1,i-radius(2)):i+radius(4),:,:),[],1);
  end
  for i = radius(2)+1:n-radius(4),
    I(i,:,:) = max(B(max(1,i-radius(2)):min(end,i+radius(4)),:,:),[],1);
  end
  for i = n-radius(4)+1:n,
    I(i,:,:) = max(B(i-radius(2):min(end,i+radius(4)),:,:),[],1);
  end
otherwise,
  error('maxfilter: poorly defined radius\n');
end
end

%% init_gabor
function [fSiz,filters,c1OL,numSimpleFilters] = init_gabor(rot, RF_siz, Div)
% function init_gabor(rot, RF_siz, Div)
% Thomas R. Serre
% Feb. 2003

c1OL             = 2;
numFilterSizes   = length(RF_siz);
numSimpleFilters = length(rot);
numFilters       = numFilterSizes*numSimpleFilters;
fSiz             = zeros(numFilters,1);	% vector with filter sizes
filters          = zeros(max(RF_siz)^2,numFilters);

lambda = RF_siz*2./Div;
sigma  = lambda.*0.8;
G      = 0.3;   % spatial aspect ratio: 0.23 < gamma < 0.92

for k = 1:numFilterSizes  
    for r = 1:numSimpleFilters
        theta     = rot(r)*pi/180;
        filtSize  = RF_siz(k);
        center    = ceil(filtSize/2);
        filtSizeL = center-1;
        filtSizeR = filtSize-filtSizeL-1;
        sigmaq    = sigma(k)^2;
        
        for i = -filtSizeL:filtSizeR
            for j = -filtSizeL:filtSizeR
                
                if ( sqrt(i^2+j^2)>filtSize/2 )
                    E = 0;
                else
                    x = i*cos(theta) - j*sin(theta);
                    y = i*sin(theta) + j*cos(theta);
                    E = exp(-(x^2+G^2*y^2)/(2*sigmaq))*cos(2*pi*x/lambda(k));
                end
                f(j+center,i+center) = E;
            end
        end
       
        f = f - sum(f);%mean(mean(f));
        f = f ./ sqrt(sum(sum(f.^2)));
        p = numSimpleFilters*(k-1) + r;
        filters(1:filtSize^2,p)=reshape(f,filtSize^2,1);
        fSiz(p)=filtSize;
    end
end
end

%% padimage
function o = padimage(i,amnt,method)
%function o = padimage(i,amnt,method)
%
%padarray which operates on only the first 2 dimensions of a 3 dimensional
%image. (of arbitrary number of layers);
%
%amnt is a scalar value indicating how many pixels to buffer each side
%with.
%
%String values for pad method
%        'circular'    Pads with circular repetion of elements.
%        'replicate'   Repeats border elements of A.
%        'symmetric'   Pads array with mirror reflections of itself. 
%
%method may also be a constant value, like 0.0
if(nargin < 3)
   method = 'replicate';
end
o = zeros(size(i,1) + 2 * amnt, size(i,2) + 2* amnt, size(i,3));
for n = 1:size(i,3)
  o(:,:,n) = padarray(i(:,:,n),[amnt,amnt],method,'both');
end
end

%% sumfilter
function I3 = sumfilter(I,radius);
%function I3 = sumfilter(I,radius);
%
%I is the input image
%radius is the additional radius of the window, i.e., 5 means 11 x 11
%if a four value vector is specified for radius, then any rectangular support may be used for max.
%in the order left top right bottom.

if (size(I,3) > 1)
    error('Only single-channel images are allowed');
end

switch length(radius)
  case 4,
    I2 = conv2(ones(1,radius(2)+radius(4)+1), ones(radius(1)+radius(3)+1,1), I);
    I3 = I2((radius(4)+1:radius(4)+size(I,1)), (radius(3)+1:radius(3)+size(I,2)));
  case 1,
    mask = ones(2*radius+1,1);
    I2 = conv2(mask, mask, I);
    I3 = I2((radius+1:radius+size(I,1)), (radius+1:radius+size(I,2)));  
end
end

%% readAllImages
function cI = readAllImages(train_set,test_set,maximagesperdir);
%Reads all training and testing images into a cell of length 4
%cI{1} = train_set.pos,
%cI{2} = train_set.neg,
%cI{3} = test_set.pos,
%cI{4} = test_set.neg,
if nargin<3
  maximagesperdir = inf;
end

dnames = {train_set.pos,train_set.neg,test_set.pos,test_set.neg};

fprintf('Reading images...');    
cI = cell(4,1);
for i = 1:4,
  c{i} = dir(dnames{i});
  if length(c{i})>0,
    if c{i}(1).name == '.',
      c{i} = c{i}(3:end);
    end
  end
  if length(c{i})>maximagesperdir,
    c{i} = c{i}(1:maximagesperdir);
  end
  cI{i} = cell(length(c{i}),1);
  for j = 1:length(c{i}),
    cI{i}{j} = double(imread([dnames{i} '/' c{i}(j).name]))./255;
  end
end

fprintf('done.\n');
end

%% unpadimage
function o = unpadimage(i,amnt)
%function o = unpadimage(i,amnt)
%
%un does padimage
%if length(amnt == 1), unpad equal on each side
%if length(amnt == 2), first amnt is left right, second up down
%if length(amnt == 4), then [left top right bottom];

switch(length(amnt))
case 1
  sx = size(i,2) - 2 * amnt;
  sy = size(i,1) - 2 * amnt;
  l = amnt + 1;
  r = size(i,2) - amnt;
  t = amnt + 1;
  b = size(i,1) - amnt;
case 2
  sx = size(i,2) - 2 * amnt(1);
  sy = size(i,1) - 2 * amnt(2);
  l = amnt(1) + 1;
  r = size(i,2) - amnt(1);
  t = amnt(2) + 1;
  b = size(i,1) - amnt(2);
case 4
  sx = size(i,2) - (amnt(1) + amnt(3));
  sy = size(i,1) - (amnt(2) + amnt(4));
  l = amnt(1) + 1;
  r = size(i,2) - amnt(3);
  t = amnt(2) + 1;
  b = size(i,1) - amnt(4);
otherwise
  error('illegal unpad amount\n');
end
if(any([sx,sy] < 1))
    fprintf('unpadimage newsize < 0, returning []\n');
    o = [];
    return;
end
o = i(t:b,l:r,:);
end

%% WindowedPatchDistance
function D = WindowedPatchDistance(Im,Patch)
%function D = WindowedPatchDistance(Im,Patch)
%
%computes the euclidean distance between Patch and all crops of Im of
%similar size.
%
% sum_over_p(W(p)-I(p))^2 is factored as
% sum_over_p(W(p)^2) + sum_over_p(I(p)^2) - 2*(W(p)*I(p));
%
% Im and Patch must have the same number of channels
%
dIm = size(Im,3);
dPatch = size(Im,3);
if(dIm ~= dPatch)
  fprintf('The patch and image must be of the same number of layers');
end
s = size(Patch);
s(3) = dIm;
Psqr = sum(sum(sum(Patch.^2)));
Imsq = Im.^2;
Imsq = sum(Imsq,3);
sum_support = [ceil(s(2)/2)-1,ceil(s(1)/2)-1,floor(s(2)/2),floor(s(1)/2)];
Imsq = sumfilter(Imsq,sum_support);
PI = zeros(size(Imsq));

for i = 1:dIm
	PI = PI + conv2(Im(:,:,i),Patch(:,:,i), 'same');
end

D = Imsq - 2 * PI + Psqr + 10^-10;



end

















