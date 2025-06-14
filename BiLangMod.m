N = 100000; t = 0.01;
a = 0.01; b = 0.01;

ABCDEBi = zeros(N,7);

c = rand +0.5; d = rand +0.5; e = rand +0.5; f = rand +0.5; g = rand +0.5;    
ABCDEBi(1,:) = [1000,1000,3000,34000,35000,0,1];
for i = 2:N
    if (rem(i,1000) == 0)
    c = rand +0.5; d = rand +0.5; e = rand +0.5; f = rand +0.5; g = rand +0.5;
    end
    ABCDEBi(i,1) = ABCDEBi(i-1,1) + c*a*ABCDEBi(i-1,1)*(ABCDEBi(i-1,6)/(75000-ABCDEBi(i-1,6)))*t - b*ABCDEBi(i-1,1)*(75000-ABCDEBi(i-1,1))^2/75000^2/c*t;
    ABCDEBi(i,2) = ABCDEBi(i-1,2) + d*a*ABCDEBi(i-1,2)*(ABCDEBi(i-1,6)/(75000-ABCDEBi(i-1,6)))*t - b*ABCDEBi(i-1,2)*(75000-ABCDEBi(i-1,2))^2/75000^2/d*t;
    ABCDEBi(i,3) = ABCDEBi(i-1,3) + e*a*ABCDEBi(i-1,3)*(ABCDEBi(i-1,6)/(75000-ABCDEBi(i-1,6)))*t - b*ABCDEBi(i-1,3)*(75000-ABCDEBi(i-1,3))^2/75000^2/e*t;
    ABCDEBi(i,4) = ABCDEBi(i-1,4) + f*a*ABCDEBi(i-1,4)*(ABCDEBi(i-1,6)/(75000-ABCDEBi(i-1,6)))*t - b*ABCDEBi(i-1,4)*(75000-ABCDEBi(i-1,4))^2/75000^2/f*t;
    ABCDEBi(i,5) = ABCDEBi(i-1,5) + g*a*ABCDEBi(i-1,5)*(ABCDEBi(i-1,6)/(75000-ABCDEBi(i-1,6)))*t - b*ABCDEBi(i-1,5)*(75000-ABCDEBi(i-1,5))^2/75000^2/g*t;
    ABCDEBi(i,6) = 75000-ABCDEBi(i,1)-ABCDEBi(i,2)-ABCDEBi(i,3)-ABCDEBi(i,4)-ABCDEBi(i,5);
    ABCDEBi(i,7) = i;
end
scatter(ABCDEBi(:,7),ABCDEBi(:,5));