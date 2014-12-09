load('mnist_all.mat');
%imshow(reshape(train7(5,:), 28, 28)');
C  = [0.01 0.1 1 2 4] ;

itrain0 = randi(length(train0),100,1);
itrain1 = randi(length(train1),1,100);
itrain2 = randi(length(train2),1,100);
itrain3 = randi(length(train3),1,100);
itrain4 = randi(length(train4),1,100);
itrain5 = randi(length(train5),1,100);
itrain6 = randi(length(train6),1,100);
itrain7 = randi(length(train7),1,100);
itrain8 = randi(length(train8),1,100);
itrain9 = randi(length(train9),1,100);

X0 = train0(itrain0,:)';
X1 = train1(itrain1,:)';
X2 = train2(itrain2,:)';
X3 = train3(itrain3,:)';
X4 = train4(itrain4,:)';
X5 = train5(itrain5,:)';
X6 = train6(itrain6,:)';
X7 = train7(itrain7,:)';
X8 = train8(itrain8,:)';
X9 = train9(itrain9,:)';

X = [X0 X1 X2 X3 X4 X5 X6 X7 X8 X9];

Y = [zeros(1,100) ones(1,100) zeros(1,800)];

for i=1:length(C)
    model = svmdemo(X, Y, 'linear', C(i)) ;
end

T = [test0' test1' test2' test3' test4' test5' test6' test7' test8' test9'];
sz0 = size(test0,1);
sz1 = size(test1,1);
szT = size(T,2);

T_gt_labels = [zeros(1,sz0) ones(1,sz1) zeros(1,szT-(sz0+sz1))];

T_predicted_labels = predict(model,T);


