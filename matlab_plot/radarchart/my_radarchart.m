f1 = figure(1);
subplot(1,2,1)
ax1 = gca;

% 
X1 = [[93.9022, 9.6903, 0.76, 5.5416];
      [83.8399, 9.1564, 0.79, 5.7876];
      [89.7806, 9.4753, 0.77, 6.5640];
      [72.6273, 8.5222, 0.82, 4.5637]];

X = X1 ./ (max(X1));

RC = radarChart(X, 'Type', 'Patch');
RC.PropName = {'MSE','RMSE','R2','MAE'};
RC.ClassName = {'RF', 'DRF', 'TRF', 'DTRF'};
RC = RC.draw();

RC.setPropLabel('FontSize', 10, 'FontName', 'Times New Roman', 'Color', [0, 0, 0]);


% 隐藏径向刻度标签
set(RC.RLabelHdl, 'Visible', 'off');

colorList = [0.40 0.76 0.60;
             0.99 0.55 0.38;
             0.55 0.63 0.80;
             0.25 0.83 0.20];
for n = 1:RC.ClassNum
    RC.setPatchN(n, 'FaceColor', colorList(n, :), 'EdgeColor', colorList(n, :));
end
ax1.Title.String         = '\fontname{Times New Roman}SHAOUTTE';
ax1.Title.Position = [0.01 1.2 -1];
ax1.Title.FontSize = 12;


subplot(1,2,2)
ax2 = gca;
% SHBOUTTE.AV
Y1 = [[97.3533, 9.8668, 0.75, 5.4406];
      [85.9328, 9.27, 0.78, 5.5283];
      [85.2364, 9.2324, 0.78,  5.9128];
      [60.9181, 7.8050, 0.84,  4.1168]];

Y = Y1 ./ (max(Y1));

RC = radarChart(Y, 'Type', 'Patch');
RC.PropName = {'MSE','RMSE','R2','MAE'};
RC.ClassName = {'RF', 'DRF', 'TRF', 'DTRF'};
RC = RC.draw();

RC.setPropLabel('FontSize', 10, 'FontName', 'Times New Roman', 'Color', [0, 0, 0]);
% 隐藏径向刻度标签
set(RC.RLabelHdl, 'Visible', 'off');

% 设置图例水平放置
RC.legend();
RC.setLegend('FontSize', 8);  % 设置字体大小为 10
RC.setLegend('Position', [0.3, 0.02, 0.4, 0.05]);  % 调整图例的位置和大小
set(RC.LgdHdl, 'Orientation', 'horizontal', 'NumColumns', RC.ClassNum);


colorList = [0.40 0.76 0.60;
             0.99 0.55 0.38;
             0.55 0.63 0.80;
             0.25 0.83 0.20];
for n = 1:RC.ClassNum
    RC.setPatchN(n, 'FaceColor', colorList(n, :), 'EdgeColor', colorList(n, :));
end
ax2.Title.String         = '\fontname{Times New Roman}SHBOUTTE';
ax2.Title.Position = [0.01 1.2 -1];
ax2.Title.FontSize = 12;

f1.Color = 'white'; % 背景颜色为白色
f1.Units = 'centimeters'; % 设置单位
f1.Position = [10 10 15 9]; % 设置图片尺寸