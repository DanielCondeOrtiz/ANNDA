function plot_data_and_decision_boundary(patterns, targets, w, plot_title, figure_number, bias)
%PLOT_DATA_AND_DECISION_BOUNDARY plots the datapoints contained in patterns
%according to their labels in targets. Moreover, a decision boundary is
%plotted according to the elements in w. The title and the figure number
%can also be given.

[class_points, class_identifiers, ~] = data_seperation(patterns, targets, bias);
class_count = length(class_points);

figure(figure_number)

colors = {'r.','b.','c.'};

for class_index = 1:class_count
    class_data_points = class_points{class_index};
    scatter(class_data_points(1,:),class_data_points(2,:),colors{class_index})
    hold on
end

if bias
    w1= ([w(1),w(2)]./norm(w))*(-w(3))/norm(w);
    w2=[w1(2),-w1(1)]+w1;
else
    w1=[-w(2), w(1)];
    w2=-w1;
end

m = (w2(2)-w2(1))/(w1(2)-w1(1));
n1 = w2(2)*m - w1(2);
y1 = m*-3 + n1;
y2 = m*3 + n1;
line([-3,3],[y1 y2])


xlim([-3 3])
ylim([-3 3])



title(plot_title)
hold off
legend_text = num2str(class_identifiers(:));
legend(legend_text)

end

