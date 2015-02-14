function visualizeWireframe(points,partNames,wireframe)

hold on;
edges = wireframe.edges;

defColor = (size(edges,2)>2);

x  = points(1,:);
y  = points(2,:);
z  = points(3,:);
plotColor = 'b';

for i = 1:size(edges,1)
  p1 = find(ismember(partNames,edges{i,1}));
  p2 = find(ismember(partNames,edges{i,2}));
  if(isempty(p2))
      keyboard;
  end
  p  = [p1 p2];
  if(defColor)
      plotColor = edges{i,3};
  end
  plot3(x(p),y(p),z(p),'Color',plotColor);
end

drawnow;
axis equal;
grid on;

end