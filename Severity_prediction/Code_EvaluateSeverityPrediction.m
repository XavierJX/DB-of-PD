clc
clear
close all
parti = [1 0 2 3];
limi = [100 100 25 25];
for i =1:4
    Pre_y_test = load(['XGboost_pretest' num2str(parti(i)) '.txt']);
    y_test = load(['ytest'  num2str(parti(i)) '.txt']);
    Pre_y_train = load(['XGboost_pre_train' num2str(parti(i)) '.txt']);
    y_train = load(['ytrain' num2str(parti(i)) '.txt']);

    dotSize = abs([y_train;y_test]-[Pre_y_train;Pre_y_test]);
    dotSize1 = 36-rescale(dotSize,1,35);

    subplot(2,2,i);
    scatter(y_train,Pre_y_train,'o','MarkerEdgeColor','none','MarkerFaceColor',[56,81,163]/255,'SizeData',dotSize1(1:size(y_train,1),1),'MarkerFaceAlpha', 0.2);
    hold on
    scatter(y_test,Pre_y_test,'o','MarkerEdgeColor','none','MarkerFaceColor',[169,2,38]/255,'SizeData',dotSize1(size(y_train,1)+1:end,1),'MarkerFaceAlpha', 0.5);
    xlim([0,limi(i)]);
    ylim([0,limi(i)]);

    R2(1,i)=1 - (sum(([Pre_y_train;Pre_y_test]- [y_train;y_test]).^2) / sum(([y_train;y_test] - mean([y_train;y_test])).^2));
    R2(2,i)=1 - (sum((Pre_y_train- y_train).^2) / sum((y_train - mean(y_train)).^2));
    R2(3,i)=1 - (sum((Pre_y_test- y_test).^2) / sum((y_train - mean(y_test)).^2));
    [Cor_spear(1,i), p_spear(1,i)]=corr([y_train;y_test],[Pre_y_train;Pre_y_test],'Type','Spearman','rows','complete');
    % [Cor_spear(2,i), p_spear(2,i)]=corr(y_train,Pre_y_train,'Type','Spearman','rows','complete');
    % [Cor_spear(3,i), p_spear(3,i)]=corr(y_test,Pre_y_test,'Type','Spearman','rows','complete');
end



