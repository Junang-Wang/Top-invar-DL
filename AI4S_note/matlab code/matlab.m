D_deter = readmatrix('D_total_cone'); 
figure(1);
mu = linspace(-3,3,100);
B = linspace(-3,3,100);
[Mu,B_p] = meshgrid(mu,B);
surf(Mu,B,D_deter)
colorbar()
xlabel('\mu')
ylabel('B')
zlabel('\pi^2|D|')
set(gca,'fontsize',14)
figure(2);
plot(B,D_deter(50,:),'LineWidth',2);
xlabel('B')
ylabel('\pi^2|D|')
set(gca,'fontsize',14)
