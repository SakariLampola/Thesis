function [o] = DetectObjects(ImageFile,Confidence)
%DETECTOBJECTS Detect objects using system call
%   Objects = table of object classes and bounding boxes
%       category : car
%       startX : 430
%       startY : 316
%       endX : 706
%       endY :494
%       confidence : 0.956794
%   ImageFile : full path to the image file
%   Confidence : level of confidence for the objects included, 0...1
    % Spawn the Python application
    command = "python detect_objects.py --image ";
    command = strcat(command,ImageFile);
    command = strcat(command," --confidence ");
    command = strcat(command,num2str(Confidence));
    [status,cmdout] = system(command);
    if status ~= 0; o=null; end
    % Parse the results
    c=textscan(cmdout,'%s %d %d %d %d %f');
    % Change the results to a struct array
    [h,~]=size(c{1});
    for i=1:h
        o(i).class=c{1}(i);
        o(i).startX=c{2}(i);
        o(i).startY=c{3}(i);
        o(i).endX=c{4}(i);
        o(i).endY=c{5}(i);
        o(i).confidence=c{6}(i);
    end
end

