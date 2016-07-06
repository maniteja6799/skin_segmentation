function [ tp,tn,fp,fn ] = predict_and_return( data ,theta )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
tp=0;tn=0;fp=0;fn=0;
for i = 1: size(data)
    %temp = data(i,1:3); 
    temp = mapFeature(data(i,1),data(i,2),data(i,3));
    if(sigmoid(temp*theta) < 0.5)
        pr=0;
    else
        pr=1;
    end
    
    if (pr==1) && data(i,4)==1
        tp = tp+1;
    end
    if pr==1 && data(i,4)==2
        fp = fp+1;
    end
    if pr==0 && data(i,4)==1
        fn = fn+1;
    end
    if pr==0 && data(i,4)==2
        tn = tn+1;
    end
end


end

