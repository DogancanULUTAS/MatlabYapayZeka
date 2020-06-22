function [net ye yv MAPE R2 MPE RMSE] = one( input, target, training_rate,n1,n2, lrate )

noofdata=size(input,1);

ntd=round(noofdata*training_rate);

xt=input(1:ntd,:);
xv=input(ntd+1:end,:);

yt=target(1:ntd);
yv=target(ntd+1:end);

xt=xt';
xv=xv';
yt=yt';
yv=yv';

xtn=mapminmax(xt);
xvn=mapminmax(xv);

[ytn, ps]=mapminmax(yt);

net=newff(xtn, ytn, [n1,n2], {}, 'trainbr' );
net.trainParam.lr=lrate;
net.trainParam.epochs=10000;
net.trainParam.goal=1e-10000000000;
net.trainParam.show=NaN; 

net=train(net,xtn,ytn);

yen=sim(net, xvn);

ye=mapminmax('reverse', yen, ps);


ye=ye';
yv=yv';

MAPE=mean((abs(ye-yv))./yv);

SStotal=sum((yv-mean(yv)).^2);
SSerror=sum((ye-yv).^2);
R2=1-SSerror/SStotal;
MPE=mean((ye+yv)./yv);
RMSE=mean((sqrt(ye-yv))./yv);

end
