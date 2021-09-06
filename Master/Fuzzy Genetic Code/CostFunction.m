function [ mse ] = CostFunction(Data, chromosome, c, sigma )
nData = size(Data,1);
e = zeros(1,nData);
    for i=1:nData
        x = Data(i,:);
        output = Infererece(x,chromosome, c, sigma);
        e(i) = (output - x(4))^2;
    end
    mse = sum(e)/nData;
end

