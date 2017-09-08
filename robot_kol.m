clear
clc
merkez=[0,0]; %robotun konumu, orjin kabul edildi
kol_uzunluk=200; %kolun toplam uzunlugu (cm)
A=360*rand(1,6); %ilk acilar, egitim verisi bu degiskende saklanir

%egitim verisi olusturma
for iterasyon=1:100 %egitim verisinin buyuklugunu belirler
    %cisim y=120 dogrusu uzerindedir ve x en fazla 160 olabilir
    % 3-4-5 ucgeni
    cisim=[(-160+(320*rand(1,1))),120]; %cismin konumu x=random,y=120
    acilar=double.empty(0,0);%bos acilar matrisi, hafiza icin
    for alfa=0:360 %servo1 acisi
        x=cosd(alfa)*(kol_uzunluk/2); %kol parca1 in konumu x
        y=sind(alfa)*(kol_uzunluk/2); %kol parca1 in konumu y
        for beta=0:360 %servo2 acisi
            x2=x + (cosd(beta)*(kol_uzunluk/2)); %kol parca2 nin konumu x
            y2=y + (sind(beta)*(kol_uzunluk/2)); %kol parca2 nin konumu y
            if fix(x2-cisim(1))==0 && fix(y2-cisim(2))==0 %kol cisime temas ediyor mu?
                acilar=[acilar;alfa,beta]; %bu acilari hafizaya ekle
            end
        end
    end
    if size(acilar,1)<2
        continue; %hic secenek hesaplayamamis, yuvarlama hatalarindan kaynaklaniyor
    elseif size(acilar,1)>2 %2den fazla secenek hesaplamis,bu da yuvarlama hatasi
        for i=1:(size(acilar,1)-1)
            if ceil(acilar(i,1))~=ceil(acilar(i+1,1)) %ayni acilari silmek icin
                acilar=acilar(i:i+1,:);
                break;
            end
        end     
    end
    if size(acilar,1)==2 %2 secenek hesaplamis, dogru yolda:D
        yol_1=abs( A(end,5)-acilar(1,1) ) + abs( A(end,6)-acilar(1,2) );
        %1.secenekte katedilen toplam yol(aci)
        yol_2=abs( A(end,5)-acilar(2,1) ) + abs( A(end,6)-acilar(2,2) );
        %2.secenekte katedilen toplam yol(aci)
        if yol_1<yol_2 %hangi secenekte az yol katedildiyse onu egitim verisine ekle
            A=[A; A(end,5), A(end,6), cisim(1,1), cisim(1,2), acilar(1,1), acilar(1,2)];
        else
            A=[A; A(end,5), A(end,6), cisim(1,1), cisim(1,2), acilar(2,1), acilar(2,2)];
        end
    end
end

A(1,:)=[]; %ilk konumu egitim matrisimizden temizledik

%burdan sonrasi yapay zeka
inputs=( A(1:end-10,1:4) )'; %giris matrisim ilk 4 sutun
targets= ( A(1:end-10,5:6) )'; %hedef matrisim son 5 ve 6. sutun
test_i=( A(end-10:end,1:4) )'; %test giris 
test_t= ( A(end-10:end,5:6) )'; %test hedef
net = feedforwardnet(4); %agim olusturuldu, gizli katman 4 noron
net = configure(net,inputs, targets); %giris-cikis katmanlari da ayarlandi
%net.layers{2}.transferFcn = 'logsig'; %transfer fonksiyonu sigmoid
net.trainParam.max_fail = 1000; %max hatali iterasyon sayisi
[net, tr]=train(net,inputs,targets); %ag egitildi

figure;
for iterasyon=1:size(test_i,2) %test sonuclari
    gercek=test_t(:,iterasyon);
    agin=sim(net,test_i(:,iterasyon));
    fprintf('olmasi gereken = %f,%f, agin sonucu=%f,%f\n',gercek,agin);
    
    dogru=0;
    if pdist([gercek,agin],'euclidean')<10 %10cm hata payi icinde ise dogrudur
        dogru=dogru+1;
    end
    
    tx1=cosd(gercek(1))*(kol_uzunluk/2);
    ty1=sind(gercek(1))*(kol_uzunluk/2);
    tx2=tx1+cosd(gercek(2))*(kol_uzunluk/2);
    ty2=ty1+sind(gercek(2))*(kol_uzunluk/2);
    tx3=cosd(agin(1))*(kol_uzunluk/2);
    ty3=sind(agin(1))*(kol_uzunluk/2);
    tx4=tx3+cosd(agin(2))*(kol_uzunluk/2);
    ty4=ty3+sind(agin(2))*(kol_uzunluk/2);
    hold on;
    ax = gca;
    ax.XAxisLocation = 'origin';
    ax.YAxisLocation = 'origin';
    ax.Box = 'off';
    xlim([-kol_uzunluk kol_uzunluk])
    ylim([-5 160])
    xL = get(gca,'XLim');
    line(xL,[120 120],'Color','g');
    plot( [0,tx1],[0,ty1],'b' );
    plot( [0,tx3],[0,ty3],'r' );
    plot( [tx1,tx2],[ty1,ty2],'b' );
    plot( [tx3,tx4],[ty3,ty4],'r' );
    legend('y = 120 dogrusu','gercek hareketler','agin tahmini hareketleri')
end

yuzde= dogru/size(test_i,2)*100;
fprintf('Agin test basarimi = %%%f\n',yuzde);