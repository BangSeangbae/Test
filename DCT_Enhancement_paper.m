% clc; clear all;
% close all;

%% Global variables %%%%%%%%%%%%%%%%
block_size = 16;
half_block = block_size/2;
maxAmplitude = 200;
alpha = 1;

GFSmoothFactor = 255*255*0.04;
window_length = 4;
AveWin=ones(2*window_length+1,2*window_length+1)/ ((2*window_length+1)*(2*window_length+1));

% for lam=1:block_size
%     
%     if( block_size==8 )
%         lambdaPlane(lam,:)=[1 1 1 1 1 2 5 7];
%     %     lambdaEdge(lam,:)=[1 1 1 2 5 3 2 1];
%         lambdaTexture(lam,:)=[1 1 1 2 3 4 6 7];
%     
%     elseif(block_size==16)
%         lambdaPlane(lam,:)  =	[1 1 1 1 1 1 2 2 2 3 3 4 4 4 4 4];
%     %     lambdaEdge(lam,:)   =	[1 1 1 1 1 2 2 5 5 4 4 3 3 2 2 2];
%         lambdaTexture(lam,:)=   [1 1 1 2 2 2 3 3 3 4 4 4 4 4 4 4];
%     end
% end
% lambdaEdge=lambdaTexture;



% [ratio_weight_plane ratio_weight_edge ratio_weight_texture]=ratio_weight_gen(block_size,0,3);


%%%%%%%% Read image and Convert it to luminance %%%%%%%%%%%%%%%
% dirr = strcat('17.tif');
dirr = strcat('C:\Users\gift\Pictures\kodim18.png');
% dirr = strcat('C:\Users\gift\Pictures\5.bmp');

Img=imread(dirr);
figure(1), imshow(Img,[]);

%% Crop Image 
% [ROI_x ROI_y] = ginput(1);

% ROI_x=ROI_x-fix(block_size/2);
% ROI_y=ROI_y-fix(block_size/2);

% ROI_ver=block_size;
% ROI_hor=block_size;

% Img = Img(ROI_y:ROI_y+ROI_ver-1,ROI_x:ROI_x+ROI_hor-1,:);

tic;
[ver hor]=size(Img(:,:,1));
[Hue Saturation Value]=rgb2hsv(Img);

R = double(Img(:,:,1)); G = double(Img(:,:,2)); B = double(Img(:,:,3));
Y  = (0.299 * R) + (0.587 * G) + (0.114 * B);
U = -(0.16874 * R) - (0.3313 * G) + (0.500 * B) + 128;
V =  (0.500 * R) - (0.4187 * G) - (0.0813 * B) + 128;
Y = round(Y);



%%%%%%%%%%%% Enhanced Values %%%%%%%%%%%%%
Yout=Y;
E1Map=zeros(ver,hor);
TexEMap=zeros(ver,hor);
DCRMap=zeros(ver,hor);
EdgeRMap=zeros(ver,hor);
TexRMap=zeros(ver,hor);
ClassificationMap=zeros(ver,hor);
ratio_weight_map = zeros(ver,hor);

DCRMap2=zeros(ver,hor);
EdgeRMap2=zeros(ver,hor);
TexRMap2=zeros(ver,hor);

DCMap=zeros(ver,hor);
LowMap=zeros(ver,hor);
MidMap=zeros(ver,hor);
HighMap=zeros(ver,hor);

ED_verMap=zeros(ver,hor);
ED_horMap=zeros(ver,hor);
EDMap=zeros(ver,hor);

CodingGainMap=zeros(ver,hor);



%% Enhancement in DCT domain
% hb=fix(35/block_size)*block_size+1;
% vb=fix(153/block_size)*block_size+1;
% for v=vb:vb+block_size
%     for h=hb:hb+block_size
for v=1:block_size:ver
    for h=1:block_size:hor          

        BlockDCT = dct2(Y(v:v+block_size-1,h:h+block_size-1));
        BlockDCTsq = BlockDCT.*BlockDCT;
        BlockDCTabs = abs(BlockDCT);
        blockdct = abs(BlockDCT);
        block_ver = block_size; block_hor = block_size;

        [E1 E2 TexE DCR EdgeR TexR DC Low_freq Mid_freq High_freq ED_ver ED_hor ED CodingGain]=energy_pattern(BlockDCTabs);

        
%         band_energy=zeros(1,block_ver+block_hor-1);
%         cul_band_energy=zeros(1,block_ver+block_hor-1);
%         for vv=1:block_ver
%             for hh=1:block_hor
%                 band_energy(vv+hh-1)=BlockDCTsq(vv,hh)+band_energy(vv+hh-1);
%             end
%         end
% 
%         lb = 1:fix(block_size/3)+1;
%         
%         ED = ( sum( band_energy(2:fix(block_ver/3)+1) ) / sum(lb) )/ ( sum( band_energy(fix(block_ver/3)+1+1:end) ) / (block_size^2-sum(lb)) +0.01);                
%         
%         for b=1:block_ver+block_hor-1
%             if b<=fix(block_ver/2)
%                 band_energy(b)=band_energy(b)/b;
%             else
%                 band_energy(b)=band_energy(b)/(block_ver+block_hor-b);
%             end
%             
%             if b==1
%                 cul_band_energy(b)=0;
%             else
%                 cul_band_energy(b)=band_energy(b)+cul_band_energy(b-1);
%             end
%         end        
        
        ED_ver = ( mean(mean(blockdct(2:fix(block_size/3)+1,:))) ) / ( mean(mean(blockdct(fix(block_size/3)+1+1:end,:)))+0.01 );
        ED_hor = ( mean(mean(blockdct(:,2:fix(block_size/3)+1))) ) / ( mean(mean(blockdct(:,fix(block_size/3)+1+1:end)))+0.01 );
        
        
        % % % Direction % % %    
        if(block_size==8)
            Vtmp=sum(sum(BlockDCTabs(1,2:8)))+0.0001;
            Htmp=sum(sum(BlockDCTabs(2:8,1)))+0.0001;
        elseif(block_size==16)
            Vtmp=sum(sum(BlockDCTabs(1:2,2:16)))+0.0001;
            Htmp=sum(sum(BlockDCTabs(2:16,1:2)))+0.0001;
        end

%         if Vtmp==Htmp
%             Vfactor(v:v+block_size-1,h:h+block_size-1)=0.5;
%             Hfactor(v:v+block_size-1,h:h+block_size-1)=0.5;
%         else
            Vfactor(v:v+block_size-1,h:h+block_size-1)=Vtmp/(Vtmp+Htmp);
            Hfactor(v:v+block_size-1,h:h+block_size-1)=Htmp/(Vtmp+Htmp);
%         end        
        
        
        E1Map(v:v+block_size-1,h:h+block_size-1)=E1;
        TexEMap(v:v+block_size-1,h:h+block_size-1)=TexE;
        
        DCRMap(v:v+block_size-1,h:h+block_size-1)=DCR;%/(DCR+EdgeR+TexR);
        EdgeRMap(v:v+block_size-1,h:h+block_size-1)=EdgeR;%/(DCR+EdgeR+TexR);
        TexRMap(v:v+block_size-1,h:h+block_size-1)=TexR;%/(DCR+EdgeR+TexR);
        
        DCMap(v:v+block_size-1,h:h+block_size-1)=DC;
        LowMap(v:v+block_size-1,h:h+block_size-1)=Low_freq;
        MidMap(v:v+block_size-1,h:h+block_size-1)=Mid_freq;
        HighMap(v:v+block_size-1,h:h+block_size-1)=High_freq;
        
        ED_verMap(v:v+block_size-1,h:h+block_size-1)=ED_ver;
        ED_horMap(v:v+block_size-1,h:h+block_size-1)=ED_hor;
        
        EDMap(v:v+block_size-1,h:h+block_size-1)=ED;
        
        CodingGainMap(v:v+block_size-1,h:h+block_size-1)=CodingGain;
        
        
        
        % Smoothing
%         if(v>block_size & h>block_size & h+block_size-1<hor)
%             DCR=( DCR+DCRMap(v-block_size,h)+DCRMap(v,h-block_size)...
%                 +DCRMap(v-block_size,h-block_size)+DCRMap(v-block_size,h+block_size-1) )/5;
%             EdgeR=( EdgeR+EdgeRMap(v-block_size,h)+EdgeRMap(v,h-block_size)...
%                 +EdgeRMap(v-block_size,h-block_size)+EdgeRMap(v-block_size,h+block_size-1) )/5;
%             TexR=( TexR+TexRMap(v-block_size,h)+TexRMap(v,h-block_size)...
%                 +TexRMap(v-block_size,h-block_size)+TexRMap(v-block_size,h+block_size-1) )/5;   
%         end
        
        
        % % % DCT Classification
%         if( TexE < 200 ) % Plain Classification  ( TexE < 250 )
%             ClassificationMap(v:v+(block_size -1),h:h+(block_size -1)) = 0;  
%             ratio_weight=ratio_weight_plane;
%         elseif( E1<-0.07*TexE+50 ) % Plain Classification  ( E1<-0.06*TexE+60 )
%             ClassificationMap(v:v+(block_size -1),h:h+(block_size -1)) = 0.1;
%             ratio_weight=ratio_weight_plane;
%         elseif( TexE<=1500 ) % if <0, It is Edge. if >=0, It is Texture.  ( TexE<=1000 )
%             if( E1>0.05*TexE-40 ) % ( E1>0.05*TexE-30 )
%                 ClassificationMap(v:v+(block_size -1),h:h+(block_size -1)) = 0.5;    % Edge                
%                 ratio_weight=(ratio_weight_edge-1).*EdgeR+1;
%             else
%                 ClassificationMap(v:v+(block_size -1),h:h+(block_size -1)) = 0.9;    % Texture                
%                 ratio_weight=(ratio_weight_texture-1).*TexR+1;
%             end
%         else
%             if( E1>35 ) %  ( E1>20 )
%                 ClassificationMap(v:v+(block_size -1),h:h+(block_size -1)) = 0.6;    % Edge
%                 ratio_weight=(ratio_weight_edge-1).*EdgeR+1;
%             else
%                 ClassificationMap(v:v+(block_size -1),h:h+(block_size -1)) = 1;    % Texture
%                 ratio_weight=(ratio_weight_texture-1).*TexR+1;
%             end
%         end

%         ratio_weight=(ratio_weight_plane*DCR + ratio_weight_edge*EdgeR + ratio_weight_texture*TexR)/(DCR+EdgeR+TexR);
%         ratio_weight = 1/(1+0.005*TexE)*ones(block_size,block_size)+0.005*TexE/(1+0.005*TexE)*ratio_weight_edge;
%         ratio_weight = 1/(1+0.005*TexE)*ratio_weight_plane+0.005*TexE/(1+0.005*TexE)*ratio_weight_texture;

        
        % % % lambda generation % % % 
%         lambda=lambda_in.*ratio_weight;
        lambda = lambda_gen(block_size,2,BlockDCTabs(1,1)/block_size,Vtmp,Htmp);
        
        mue = BlockDCTabs(1,1)/block_size;
        Gver = Vtmp;
        Ghor = Htmp;
        

        % % % Enhancement % % %
        dctVerY = BlockDCTsq;
        lamdaVer=lambda;
        if( ED_ver<2 )
            lamdaVer(1:fix(block_size/5)+1,:)=1;
%             lamdaVer(fix(block_size/5)+2:end,:)=0;
        elseif( ED_ver<5 )
            lamdaVer(1:fix(block_size/3)+1,:)=1;
%             lamdaVer(fix(block_size/3)+2:end,:)=0;
        else
            lamdaVer(1:fix(block_size/2)+1,:)=1;
%             lamdaVer(fix(block_size/2)+2:end,:)=0;
        end

        RVer=zeros(1,block_size);
        dctVerY(1,:) = sqrt(1.*lamdaVer(1,:)).*BlockDCT(1,:);   
        for vv = 2:block_size  % u is index
            RVer(vv) =  mean( mean( dctVerY(1:vv-1,:).^2 ) )/ mean( mean( BlockDCTsq(1:vv-1,:) ) );
            dctVerY(vv,:) = sqrt(RVer(vv).*lamdaVer(vv,:)).*BlockDCT(vv,:);          
        end
        

        dctHorY = BlockDCTsq;
        lamdaHor=lambda';
        if( ED_hor<2 )
            lamdaHor(:,1:fix(block_size/5)+1)=1;
%             lamdaHor(:,fix(block_size/5)+2:end)=0;
        elseif( ED_hor<5 )
            lamdaHor(:,1:fix(block_size/3)+1)=1;
%             lamdaHor(:,fix(block_size/3)+2:end)=0;
        else
            lamdaHor(:,1:fix(block_size/2)+1)=1;
%             lamdaHor(:,fix(block_size/2)+2:end)=0;
        end

        RHor=zeros(1,block_size);
        dctHorY(:, 1) = sqrt(1.*lamdaHor(:,1)).*BlockDCT(:,1);
        for hh = 2:block_size % v is index 
            RHor(hh) = mean( mean( dctVerY(:,1:hh-1).^2 ) )/ mean( mean( BlockDCTsq(:,1:hh-1) ) );           
            dctHorY(:, hh) = sqrt(RHor(hh).*lamdaHor(:,hh)).*BlockDCT(:,hh);
        end      
                  

        lambda_re=Hfactor(v,h)*lamdaHor + Vfactor(v,h)*lamdaVer;
%         BlockDCTEnh = BlockDCT.*lambda_re;        
        BlockDCTEnh = Hfactor(v,h)*dctHorY + Vfactor(v,h)*dctVerY;


        Yout(v:v+block_size-1,h:h+block_size-1)=idct2(BlockDCTEnh);
        
        
        BlockDCTsq_tmp = BlockDCTsq;
        BlockDCTsq_tmp(1,1)=0;
        cn_ver(1)=0;
        cn_hor(1)=0;
        for vv=2:block_size
            cn_ver(vv) = mean( mean( BlockDCTsq(vv,:) ) )/ mean( mean( BlockDCTsq(1:vv-1,:) ) );
            cn_hor(vv) = mean( mean( BlockDCTsq(:,vv) ) )/ mean( mean( BlockDCTsq(:,1:vv-1) ) );
        end
        band_energy=zeros(1,block_ver+block_hor-1);
        band_num=zeros(1,block_ver+block_hor-1);
        for vv=1:block_ver
            for hh=1:block_hor
                band_energy(vv+hh-1) = blockdct(vv,hh)+band_energy(vv+hh-1);
                band_num(vv+hh-1) = blockdct(vv,hh)/blockdct(vv,hh)+band_num(vv+hh-1);
            end
        end
        
        cn = [];
        for b=2:block_ver+block_hor-1
            cn(b-1)=( band_energy(b)/band_num(b) ) / ( sum(band_energy(1:b-1))/sum(band_num(1:b-1)) );
        end        

    end
end

R = Yout + 1.402*(V - 128);
G = Yout - 0.34414*(U - 128) - 0.71414*(V - 128);
B = Yout + 1.772*(U - 128);
ImgEnh = uint8(cat(3,R, G, B));

toc;


% lambda_show=zeros(2*block_size,2*block_size);
% for lv=-7:8
%     for lh=-7:8
%         
%         if( lv>0 && lh>0 )
%             lambda_show(lv+block_size,lh+block_size)=lambda_re(lv,lh);
%         elseif( lv>0 && lh<=0 )
%             lambda_show(lv+block_size,lh+block_size)=lambda_re(lv,abs(lh)+1);
%         elseif( lv<=0 && lh>0 )
%             lambda_show(lv+block_size,lh+block_size)=lambda_re(abs(lv)+1,lh);
%         else
%             lambda_show(lv+block_size,lh+block_size)=lambda_re(abs(lv)+1,abs(lh)+1);
%         end
%         
%     end
% end


% figure(1), imshow(EdgeRMap,[0 1])
% figure(2), imshow(TexRMap,[0 1])
% figure(3), imshow(DCRMap,[0 1])
% figure, imshow(ClassificationMap,[]);
% figure(4),imshow(im2uint8(Img));
figure,imshow(im2uint8(ImgEnh));
%     dirr = strcat('C:\Users\gift\Desktop\Yout.tif');
%     imwrite(ImgEnh ,dirr);

    
% v=835
% h=124
% Freq = [DCMap(v,h),LowMap(v,h),MidMap(v,h),HighMap(v,h)]
% figure, stem(Freq); axis([0 5 0 1000]);

% FreqR = [DCRMap(v,h),EdgeRMap(v,h),TexRMap(v,h)]
% figure, stem(FreqR); axis([0 4 0 100]);

% fftshift(fft2(Y(v:v+block_size-1,h:h+block_size-1)))

% figure, imshow(Y(vb:vb+block_size-1,hb:hb+block_size-1),[0 255])
% figure, plot(cn); axis([0 33 0 0.5])
%% surf로 보여주는 코드
% % hb=fix(172/8)*8+1;
% % vb=fix(100/8)*8+1;
% % Signal_spectrum = abs(fftshift(fft2(Y(v:v+block_size-1,h:h+block_size-1))));
% % Signalout_spectrum = abs(fftshift(fft2(Yout(v:v+block_size-1,h:h+block_size-1))));
% Signal_spectrum = abs((dct2(Y(vb:vb+block_size-1,hb:hb+block_size-1))));
% Signalout_spectrum = abs((dct2(Yout(vb:vb+block_size-1,hb:hb+block_size-1))));
% 
% 
% figure, surf( 2*(Signal_spectrum).^0.4,'EdgeColor','none'); axis([0 20 0 20 0 70]);
% colormap([108 123 139]/255); title('intput');
% shading interp; lightangle(100,30); set(gcf,'Renderer','zbuffer');
% set(findobj(gca,'type','surface'),...
%     'FaceLighting','phong',...
%     'AmbientStrength',.3,'DiffuseStrength',.8,...
%     'SpecularStrength',.9,'SpecularExponent',25,...
%     'BackFaceLighting','unlit');
% grid off; axis off; view([120 35 150]);
% 
% 
% 
% % base_dct_block_tmp = dct2(Y(v:v+block_size-1,h:h+block_size-1));
% % for v=1:block_size
% %     for h=1:block_size
% %         if( v+h-1 > round(block_size/3-(1-EDMap)) )
% %             base_dct_block_tmp(v,h) = 0;
% %         end
% %     end
% % end
% % base = idct2(base_dct_block_tmp);
% % Signal_spectrum_base = abs(fftshift(fft2(base)));
% % 
% % figure, surf( (Signal_spectrum_base).^0.3 ,'EdgeColor','none'); axis([0 20 0 20 0 30]);
% % colormap([108 123 139]/255); title('intput base');
% % shading interp; lightangle(100,30); set(gcf,'Renderer','zbuffer');
% % set(findobj(gca,'type','surface'),...
% %     'FaceLighting','phong',...
% %     'AmbientStrength',.3,'DiffuseStrength',.8,...
% %     'SpecularStrength',.9,'SpecularExponent',25,...
% %     'BackFaceLighting','unlit');
% % grid off; axis off; view([20 30 60]);
% 
% 
% 
% % detail_dct_block_tmp = dct2(Y(v:v+block_size-1,h:h+block_size-1));
% % for v=1:block_size
% %     for h=1:block_size
% %         if( v+h-1 <= round(block_size/3-(1-EDMap(v,h)))+1 )
% %             detail_dct_block_tmp(v,h) = 0;
% %         end
% %     end
% % end
% % detail = idct2(detail_dct_block_tmp);
% % Signal_spectrum_detail = abs(fftshift(fft2(detail)));
% % % Signal_spectrum_detail(1,1)=100000;
% % figure, surf( (Signal_spectrum_detail+500).^0.3 ,'EdgeColor','none'); axis([0 20 0 20 5 8]);
% % colormap([108 123 139]/255); title('intput detail');
% % shading interp; lightangle(100,30); set(gcf,'Renderer','zbuffer');
% % set(findobj(gca,'type','surface'),...
% %     'FaceLighting','phong',...
% %     'AmbientStrength',.3,'DiffuseStrength',.8,...
% %     'SpecularStrength',.9,'SpecularExponent',25,...
% %     'BackFaceLighting','unlit');
% % grid off; axis off; view([20 30 60]);
% 
% 
% figure, surf( 2*(Signalout_spectrum).^0.4 ,'EdgeColor','none'); axis([0 20 0 20 0 70]);
% colormap([108 123 139]/255); title('output');
% shading interp; lightangle(100,30); set(gcf,'Renderer','zbuffer');
% set(findobj(gca,'type','surface'),...
%     'FaceLighting','phong',...
%     'AmbientStrength',.3,'DiffuseStrength',.8,...
%     'SpecularStrength',.9,'SpecularExponent',25,...
%     'BackFaceLighting','unlit');
% grid off; axis off; view([120 35 150]);
% % 
% % 
% % 
% % figure, surf( (Signalout_spectrum./Signal_spectrum) ,'EdgeColor','none');
% % colormap([108 123 139]/255); title('lambda');
% % 
% % shading interp
% % lightangle(100,30)
% % set(gcf,'Renderer','zbuffer')
% % set(findobj(gca,'type','surface'),...
% %     'FaceLighting','phong',...
% %     'AmbientStrength',.3,'DiffuseStrength',.8,...
% %     'SpecularStrength',.9,'SpecularExponent',25,...
% %     'BackFaceLighting','unlit')
% % grid off; axis off
% % view([20 40 30]);

















% DCMap(v,h) / ( abs(LowMap(v,h)-MidMap(v,h))+abs(HighMap(v,h)-MidMap(v,h))+abs(HighMap(v,h)-LowMap(v,h)) )

% if( TexE < 200 ) % Plain Classification  ( TexE < 250 )
%     ClassificationMap(v:v+(block_size -1),h:h+(block_size -1)) = 0;
%     lambda=lambdaPlane;
%     ratio_weight=ratio_weight_plane;
% elseif( E1<-0.07*TexE+50 ) % Plain Classification  ( E1<-0.06*TexE+60 )
%     ClassificationMap(v:v+(block_size -1),h:h+(block_size -1)) = 0.1;
%     lambda=lambdaPlane;
%     ratio_weight=ratio_weight_plane;
% elseif( TexE<=1500 ) % if <0, It is Edge. if >=0, It is Texture.  ( TexE<=1000 )
%     if( E1>0.05*TexE-40 ) % ( E1>0.05*TexE-30 )
%         ClassificationMap(v:v+(block_size -1),h:h+(block_size -1)) = 0.5;    % Edge
%         lambda=lambdaEdge;
%         ratio_weight=ratio_weight_edge;
%     else
%         ClassificationMap(v:v+(block_size -1),h:h+(block_size -1)) = 0.9;    % Texture
%         lambda=lambdaTexture;
%         ratio_weight=ratio_weight_texture;
%     end
% else
%     if( E1>35 ) %  ( E1>20 )
%         ClassificationMap(v:v+(block_size -1),h:h+(block_size -1)) = 0.6;    % Edge
%         lambda=lambdaEdge;
%         ratio_weight=ratio_weight_edge;
%     else
%         ClassificationMap(v:v+(block_size -1),h:h+(block_size -1)) = 1;    % Texture
%         lambda=lambdaTexture;
%         ratio_weight=ratio_weight_texture;
%     end
% end   

% 
% OriEnergyVer = sum(sum((BlockDCTsq(:,1:1))));
% EnEnergyVer = OriEnergyVer;
% dctVerY = BlockDCTsq;
% lamdaVer=lambda';
% 
% RVer=zeros(1,block_size);
% for vv = 2:block_size  % u is index
%     RVer(vv) = EnEnergyVer/OriEnergyVer;
%     dctVerY(vv,:) = sqrt(RVer(vv).*lamdaVer(vv,:)).*BlockDCT(vv,:);    
%     OriEnergyVer = OriEnergyVer + sum( sum( BlockDCTsq(vv,:) ) );
%     EnEnergyVer = EnEnergyVer + sum( sum( dctVerY(vv,:) ) );            
% end
% 
% OriEnergyHor = sum(sum((BlockDCTsq(1:1,:)).^2));
% EnEnergyHor = OriEnergyHor;
% dctHorY = BlockDCTsq;
% lamdaHor=lambda;
% 
% RHor=zeros(1,block_size);
% for hh = 2:block_size % v is index 
%     RHor(hh) = EnEnergyHor/OriEnergyHor;            
%     dctHorY(:, hh) = sqrt(RHor(hh).*lamdaHor(:,hh)).*BlockDCT(:,hh);
%     OriEnergyHor = OriEnergyHor + sum( sum( BlockDCTsq(:, hh) ) );
%     EnEnergyHor = EnEnergyHor + sum( sum( dctHorY(:, hh) ) );
% end   
