N = 50000; t = 0.001;
a = 0.01; b = 0.05;

ABCDEBi = zeros(N,7);

ABCDEBi(1,:) = [390/4191*100,990/4191*100,345/4191*100,484/4191*100,74/4191*100,1908/4191*100,t+2025];
for i = 2:N
    ABCDEBi(i,1) = ABCDEBi(i-1,1) + a*ABCDEBi(i-1,1)*ABCDEBi(i-1,6)/(100-ABCDEBi(i-1,6))*t - b*ABCDEBi(i-1,1)*(100-ABCDEBi(i-1,1))^2/100^2*t;
    ABCDEBi(i,2) = ABCDEBi(i-1,2) + a*ABCDEBi(i-1,2)*ABCDEBi(i-1,6)/(100-ABCDEBi(i-1,6))*t - b*ABCDEBi(i-1,2)*(100-ABCDEBi(i-1,2))^2/100^2*t;
    ABCDEBi(i,3) = ABCDEBi(i-1,3) + a*ABCDEBi(i-1,3)*ABCDEBi(i-1,6)/(100-ABCDEBi(i-1,6))*t - b*ABCDEBi(i-1,3)*(100-ABCDEBi(i-1,3))^2/100^2*t;
    ABCDEBi(i,4) = ABCDEBi(i-1,4) + a*ABCDEBi(i-1,4)*ABCDEBi(i-1,6)/(100-ABCDEBi(i-1,6))*t - b*ABCDEBi(i-1,4)*(100-ABCDEBi(i-1,4))^2/100^2*t;
    ABCDEBi(i,5) = ABCDEBi(i-1,5) + a*ABCDEBi(i-1,5)*ABCDEBi(i-1,6)/(100-ABCDEBi(i-1,6))*t - b*ABCDEBi(i-1,5)*(100-ABCDEBi(i-1,5))^2/100^2*t;
    ABCDEBi(i,6) = 100-ABCDEBi(i,1)-ABCDEBi(i,2)-ABCDEBi(i,3)-ABCDEBi(i,4)-ABCDEBi(i,5);
    ABCDEBi(i,7) = i*t+2025;
end
hold on;
scatter(ABCDEBi(:,7),ABCDEBi(:,6),1, "black", "filled");
scatter(ABCDEBi(:,7),ABCDEBi(:,1),1, "red", "filled");
scatter(ABCDEBi(:,7),ABCDEBi(:,2),1, "green", "filled");
scatter(ABCDEBi(:,7),ABCDEBi(:,3),1, "blue", "filled");
scatter(ABCDEBi(:,7),ABCDEBi(:,4),1, "cyan", "filled");
scatter(ABCDEBi(:,7),ABCDEBi(:,5),1, "magenta", "filled");
xlabel('Years'); 
ylabel('Percentage of People');
legend('Multilingual', 'English', 'Mandarin', 'Hindi', 'Spanish', 'French');