function [class_points, class_identifiers, index] = data_seperation(patterns, targets, bias)
%DATA_SEPERATION seperates the data in patterns according to the classes
%specified in targets

[~, index] = sort(targets);
[class_identifiers, class_first_element] = unique(targets(index));
class_first_element(end+1) = length(targets)+1;
class_count = length(class_identifiers);
elements_per_class = zeros(class_count,1);

if bias
    patterns = patterns(1:end-1,:);
end

class_points = cell(class_count,1);

for class_index = 1:class_count
    elements_per_class(class_index) = class_first_element(class_index+1) - class_first_element(class_index);
    class_points{class_index} = patterns(:,index(class_first_element(class_index):(class_first_element(class_index)+elements_per_class(class_index)-1)));
end


end

