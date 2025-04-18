%CW1 Avi-Niam Popat
% Reference https://uk.mathworks.com/help/uav/ug/map-environment-motion-planning-using-uav-lidar.html
close all
close all hidden 

Battery_Capacity = 100;
Drain_Rate = 1; % Drain per second
battery_level = Battery_Capacity; % Initialise battery level

simTime = 60;    % in seconds
updateRate = 2;  % in Hz so update every .5 seconds
scene = uavScenario("UpdateRate",updateRate,"StopTime",simTime); 

% Floor 
addMesh(scene, "Polygon",{[0 0;80 0;80 80;40 80;40 40;0 40],[-1 0]},[0.3 0.3 0.3]); 

% Features 
addMesh(scene, "Polygon",{[10 0; 25 0; 25 15; 30 15; 30 20; 15 20; 10 10],[0 30]},[0.4660 0.6740 0.1880]); % Damaged Building 1
addMesh(scene, "Polygon",{[45 0; 80 0; 80 30; 60 30; 60 15; 45 15],[0 60]},[0.9290 0.6980 0.1250]);% Damaged Building
addMesh(scene, "Polygon",{[0 35;10 35;10 40;0 40],[0 5]},[0 0.5 0]); % Generator Room  
addMesh(scene, "Polygon",{[50 40; 70 40; 65 50; 60 45; 55 50; 50 45],[0 5]},[0 0.4470 0.7410]); % Destroyed Swimming Pool
addMesh(scene, "Polygon", {[50 40; 52 42; 51 45; 48 43], [0 2]}, [0.5 0.5 0.5]); %Pool Debris
addMesh(scene, "Polygon",{[0 0; 2 0; 2.5 4; 0.5 4],[0 3]},[0.6350 0.0780 0.1840]);% Security Room 

%Display scene
show3D(scene);
axis equal 
view([-115 20]) %camera angle

% Waypoints 
x = -20:80; 
y = -20:80; 
z = 100*ones(1,length(x)); %stays at 100m altitude

waypoints = [x' y' z']; 

orientation_eul = [0 0 0];
orientation_quat = quaternion(eul2quat(orientation_eul)); 
orientation_vec = repmat(orientation_quat,length(x),1);

time = 0:(simTime/(length(x)-1)):simTime;

trajectory = waypointTrajectory("Waypoints",waypoints,"Orientation",orientation_vec, ...
    "SampleRate",updateRate,"ReferenceFrame","ENU","TimeOfArrival",time);

plat = uavPlatform("UAV",scene,"Trajectory",trajectory,"ReferenceFrame","ENU");

updateMesh(plat,"quadrotor",{4},[1 0 0],eye(4)); 

lidarmodel = uavLidarPointCloudGenerator("AzimuthResolution",0.6, ...
    "ElevationLimits",[-90 -20],"ElevationResolution",2.5, ...
    "MaxRange",200,"UpdateRate",2,"HasOrganizedOutput",true, HasNoise=1); % set up lidar model with noise

lidar = uavSensor("Lidar",plat,lidarmodel,"MountingLocation",[0 0 -1],"MountingAngles",[0 0 0]); %attach Lidar to underside of drone
initial_pose = [-20 -20 100 1 0 0 0]; %start position & orientation of drone

[ax,plotFrames] = show3D(scene);
xlim([-15 80]);
ylim([-15 80]);
zlim([0 80]);
view([-115 20]); 
axis equal 
hold on

colormap('jet');
ptc = pointCloud(nan(1,1,3));
scatterplot = scatter3(nan,nan,nan,1,[0.3020 0.7451 0.9333], "Parent",plotFrames.UAV.Lidar);
scatterplot.XDataSource = "reshape(ptc.Location(:,:,1), [], 1)";
scatterplot.YDataSource = "reshape(ptc.Location(:,:,2), [], 1)";
scatterplot.ZDataSource = "reshape(ptc.Location(:,:,3), [], 1)";
scatterplot.CDataSource = "reshape(ptc.Location(:,:,3), [], 1) - min(reshape(ptc.Location(:,:,3), [], 1))";
hold off; 

lidarSampleTime = [];
pt = cell(1,((updateRate*simTime) +1)); 
ptOut = cell(1,((updateRate*simTime) +1)); 

map3D = occupancyMap3D(1);

setup(scene); 

ptIdx = 0;
while scene.IsRunning
    ptIdx = ptIdx + 1;
    
    % Battery monitoring
    battery_level = max(0, battery_level - Drain_Rate/updateRate); %-2 every update
    if battery_level <= 0
        disp('Battery depleted! Landing sequence initiated.');
        break;
    end
    
    % Progress update
    if mod(ptIdx, 5) == 0  % Update every 5 iterations to avoid spamming
        fprintf('Progress: %2.1f%%, Battery: %3.0f%%\n', ...
               (scene.CurrentTime/scene.StopTime)*100, ...
               battery_level);
    end
    
    % Read the simulated lidar data from the scenario
    [isUpdated,lidarSampleTime,pt{ptIdx}] = read(lidar);

    if isUpdated
        % Get Lidar sensor's pose relative to ENU reference frame.
        sensorPose = getTransform(scene.TransformTree, "ENU","UAV/Lidar",lidarSampleTime);
        % Process the simulated Lidar pointcloud.
        ptc = pt{ptIdx};
        ptOut{ptIdx} = removeInvalidPoints(pt{ptIdx});
        % Construct the occupancy map using Lidar readings.
        insertPointCloud(map3D,[sensorPose(1:3,4)' tform2quat(sensorPose)],ptOut{ptIdx},500);

        figure(1)
        show3D(scene,"Time",lidarSampleTime,"FastUpdate",true,"Parent",ax);
        xlim([-15 80]);
        ylim([-15 80]);
        zlim([0 110]);
        view([-110 20]);
        
        refreshdata
        drawnow limitrate
    end
    
    % Show map building real time 
    figure(2)
    show(map3D);
    view([-115 20]);
    axis equal 
    
    advance(scene);
    updateSensors(scene); 
    
end

map3D.FreeThreshold = 0.4;
map3D.OccupiedThreshold = 0.7; 

ss = stateSpaceSE3([0 80;0 40;0 120;inf inf;inf inf;inf inf;inf inf]);
sv = validatorOccupancyMap3D(ss);
sv.Map = map3D; 
sv.ValidationDistance = 0.1; 

planner = plannerRRTStar(ss,sv);
planner.MaxConnectionDistance = 20;
planner.ContinueAfterGoalReached = true; 
planner.MaxIterations = 500; 

planner.GoalReachedFcn = @(~,x,y)(norm(x(1:3)-y(1:3))<1);
planner.GoalBias = 0.1;


start = [3 5 5 1 0 0 0];
goal = [60 10 65 1 0 0 0]; 

rng(1,"twister"); % For repeatable results
[pthObj,solnInfo] = plan(planner,start,goal);

close all
close all hidden 

show(map3D)
axis equal
view([-115 20])
hold on
scatter3(start(1,1),start(1,2),start(1,3),'g','filled')                             % draw start state
scatter3(goal(1,1),goal(1,2),goal(1,3),'r','filled')                                % draw goal state
plot3(pthObj.States(:,1),pthObj.States(:,2),pthObj.States(:,3),'r-','LineWidth',2)  % draw path

% medical drone path
hold on;  
plot3(pthObj.States(:,1), pthObj.States(:,2), pthObj.States(:,3)-5, 'b--', 'LineWidth',1.5);  
legend('Scout Drone Path', 'Medical Drone Path');  