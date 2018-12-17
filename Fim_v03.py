import numpy as np
from scipy.ndimage import filters
import matplotlib.pyplot as plt
from skimage import io,feature
from skimage.color import rgb2gray,rgb2lab
from scipy import interpolate
from skimage.restoration import denoise_bilateral
import os
import math
from skimage.segmentation import slic,quickshift
from mpl_toolkits.mplot3d import Axes3D
import time
import cv2
from scipy.ndimage.interpolation import rotate
import scipy
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
from scipy.stats import mode
from skimage.filters import sobel

def func(imd,ime):
    im1 = rgb2gray(imd)
    imr1 = create_harris_matrix(im1)
    imsave = np.array((imr1-imr1.min())/imr1.max()*255,dtype=np.uint8)
#    io.imsave(pathsave+'\\harris'+'.png',imsave,quality=100)
    coords1 = Extrator_de_pontos_harris(imr1)
    coordssave = coords1
   
    im2 = rgb2gray(ime)
    imr2 = create_harris_matrix(im2)
    coords2 = Extrator_de_pontos_harris(imr2)
    coordssave2 = coords2
    print('Numero de pontos selecionados pro matching: ',np.shape(coords1),np.shape(coords2))
    carac1 = captura_de_caracteristicas(imd,coords1)
    carac2 = captura_de_caracteristicas(ime,coords2)
    time_start = time.clock()
    coord1 = casamento(carac1,carac2,coords1,coords2)
    coord2 = casamento(carac2,carac1,coords2,coords1)
    
    coord = casamento_duplo(coord1,coord2)
    print('Tempo gasto com correspondencia: ',time.clock()-time_start,'s')
    idx = np.where(coord>0)
    print(np.shape(coord))
    coord = coord[np.where(coord>0)]
    print(np.shape(coord))
#    print(np.shape(coord1),np.shape(coord2),np.shape(coord2))
#    print(np.shape(coords1))
    coords1 = np.array(coords1)
    coords1 = coords1[idx]
    coords2 = np.array(coords2)
    coords2 = coords2[coord]
#    print(coord1[idx],coord2[coord])
    return imr1,coords1,coords2,coordssave,coordssave2

def create_harris_matrix (im, sigma = 1):
    #Para a criação da matriz de harris é necessario a criação da matriz M1, está é obtida
    #através das derivadas Ix e Iy da matriz im, após isso, a matriz de harris é apenas 
    # Mh = Det(M1)/Trace(M1)^2
    im = rgb2gray(im)
#    im = denoise_bilateral(im, multichannel=False)
#    im = my_filter(im,1)
    #Criando as matrizes de derivadas
    imx = np.zeros(im.shape)
    imy = np.zeros(im.shape)
    filters.gaussian_filter(im,(sigma,sigma),(1,0),imx)
    filters.gaussian_filter(im,(sigma,sigma),(0,1),imy)
    
    #Calculando os componentes da matriz de harris
    Wxx = filters.gaussian_filter(imx*imx,sigma)
    Wxy = filters.gaussian_filter(imx*imy,sigma) 
    Wyy = filters.gaussian_filter(imy*imy,sigma)
    
    #Calculo do determinante e traço
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy +0.000000001
#    print(Wxx.min(),Wyy.min())
#    print((Wdet/Wtr).max(), (Wdet/Wtr).min())
    
    return (Wdet/Wtr)
def casamento2  (desc1,desc2,p1,p2,thresold = .90):
    #Encontra o melhor ponto de caracteristica da primeira imagem na segunda
    
    n = len(desc1[0])
    pos_y = np.zeros(len(desc2))
    pos_x = np.zeros(len(desc2))
    for j in range(len(desc2)):
        pos_y[j] = p2[j][0]
        pos_x[j] = p2[j][1]
    d = -np.ones((len(desc1),len(desc2)))
    for i in range(len(desc1)-1):
#        Aux = pos_y.copy()
#        print(p1[i][1],np.shape(pos_y))
#        print(i,np.shape(p1)[0], len(desc1))
#        Aux = pos_y/(p1[i][0]+1)
        Aux = np.power((pos_y-p1[i][0]),2)
#        print(Aux)
        Aux2 = pos_x - p1[i][1] + .05*len(desc1)
#        print(len(desc1))
        Aux[Aux2 < 0] = 0
        Aux[Aux2 > 600] = 0
#        Aux[pos_x < p1[i][1]] = 
#        Aux[Aux<0.97] = 0
#        Aux[Aux>1.03] = 0
        Aux[Aux>5]=0
#        Aux[Aux<-0.05]=0
        idx = np.argsort(-Aux)
        j = 0
#        print(np.shape(Aux))
        while(Aux[idx[j]]!=0):
#        for j in range(len(desc2)):
#            print(idx[j])
            j = j + 1
            d1 = (desc1[i] - np.mean(desc1[i]))/np.std(desc1[i])
            if(np.std(desc2[idx[j]])<10):
#                print('s')
                d2 = (desc2[idx[j]] - np.mean(desc2[idx[j]]))/1000000000000000000000
            else:
                d2 = (desc2[idx[j]] - np.mean(desc2[idx[j]]))/(np.std(desc2[idx[j]])+0.00000000001)
            ncc_value = np.sum(d1*d2)/(n-1)
#            print(np.shape(p2), len(desc2))
            if (ncc_value > thresold ):
#                print(ncc_value)
                d[i,idx[j]] = ncc_value
#        break
#        print(j)
    ndx = np.argsort(-d)
#    print(d.shape, ndx.shape)
    casamento = ndx[:,0]
    a = np.sort(-d)
#    print(np.shape(a))
    s = (a[:, 1]- a[:, 0]) / a[:, 0]
    idx = np.where(s>-.15)
#    print(s.shape)
    casamento[idx] = -1
    return casamento
def Extrator_de_pontos_harris(Matriz_harris, min_dist = 5 , thresold = 0.001):
    #Com a entrada dos parametros métricos e a matriz de harris, com está função será 
    #possivel a aquisição de numeros de pontos variados, quanto menor o thresold mais 
    #pontos serão retornados.
#    print('media: ',np.mean(Matriz_harris))
    #procurando os candidatos a cantos utilizando a margem do thresold
    corner_thresold = Matriz_harris.max() * thresold
    Matriz_pontos = (Matriz_harris > corner_thresold)*1
    
    #Pegando as coordenadas dos candidatos
    coords = np.array(np.nonzero(Matriz_pontos)).T
    
    #Pegando os valores dos candidatos
    Valor_coords = [Matriz_harris[c[0],c[1]] for c in coords]
    
    #Ordenando os candidatos por valor e agrupando pelos seus indexadores
    
    index_sort = np.argsort(Valor_coords)
    
    
    #Selecionar pontos permitidos a entrar na seleção
    Aloc = np.zeros(Matriz_harris.shape)
    Aloc[min_dist:-min_dist,min_dist:-min_dist] = 1
        
    #Seleciona o melhor ponto em torno da distancia minima
    coord_filt = []
    for i in index_sort:
        if Aloc[coords[i,0],coords[i,1]] == 1 :
            coord_filt.append(coords[i])
            Aloc[(coords[i,0]-min_dist):(coords[i,0]+min_dist),(coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0
    
    return coord_filt
def captura_de_caracteristicas (image, pontos_harris, wid = 5):
    #Retorna o vetor caracteristica para os pontos selecionados pelo método de harris
    X,Y,z = np.shape(image)
#    image[:,:,0] = median(image[:,:,0],np.ones((10,10)))
#    image[:,:,1] = median(image[:,:,1],np.ones((10,10)))
#    image[:,:,2] = median(image[:,:,2],np.ones((10,10)))
#    image[:,:,0] = sobel(image[:,:,0])*255+image[:,:,0]
#    image[:,:,1] = sobel(image[:,:,1])*255+image[:,:,1]
#    image[:,:,2] = sobel(image[:,:,2])*255+image[:,:,2]
#    image = rgb2gray(image)
#    image = image[:,:,1:3]
#    print(type(image))
    descritores = []
    for p in pontos_harris:
        if(p[0]<wid or p[1]<wid or p[0]+wid+1>X or p[1]+wid+1>Y):
            x = 0
        else:
            pacote = image[p[0]-wid:p[0]+wid, p[1]-wid:p[1]+wid].flatten()
#        pacote = np.append(pacote,np.array([p[0],p[1]]))
            descritores.append(pacote)
    return descritores

def casamento  (desc1,desc2,p1,p2,thresold = .9):
    #Encontra o melhor ponto de caracteristica da primeira imagem na segunda
        
    n = len(desc1[0])
    d = -np.ones((len(desc1),len(desc2)))
    desc1 = np.array(desc1)
    desc2 = np.array(desc2)
    stda = desc1.std(axis=1)
    stda = (np.ones(desc1.shape)*np.expand_dims(stda,1))
    stdb = desc2.std(axis=1)
    stdb = (np.ones(desc2.shape)*np.expand_dims(stdb,1))
    meana = desc1.mean(axis=1)
    meana = (np.expand_dims(meana,1)*np.ones(desc1.shape))
    meanb = desc2.mean(axis=1)
    meanb = (np.expand_dims(meanb,1)*np.ones(desc2.shape))
    a = (desc1-meana)/stda
    b = (desc2-meanb)/stdb

    d = (a@b.T)/(n-1)
    d = np.where(d>thresold,d,-1)
#    for i in range(len(desc1)):
#        for j in range(len(desc2)):
#            d1 = (desc1[i]-np.mean(desc1[i]))/np.std(desc1[i])
#            d2 = (desc2[j]-np.mean(desc2[j]))/np.std(desc2[j])
#            ncc_value = np.sum(d1*d2)/(n-1)
#            if(ncc_value>thresold):
#                d[i,j]=ncc_value
    ndx = np.argsort(-d)
    casamento = ndx[:,0]
    return casamento
def casamento_duplo(casamento1,casamento2):
    ndx = np.where(casamento1>=0)[0]
    #removendo os casamentos não simetricos
#    print(ndx)
    for n in (ndx):
        if casamento2[casamento1[n]] != n:
            casamento1[n] = -1
    return casamento1

def disp_map(im, coords1,coords2):
    X,Y,Z = np.shape(im)
    R = np.zeros((X,Y)) - 1 
    i = 0
    desprezados = 0
    for coord in coords1:
#        print(coord[0],coord[1],coord)
#        s = coords2[i,0]-coords1[i,0]
#        if((coords2[i,1] - coords1[i,1])>0 and (coords2[i,1] - coords1[i,1])<35 ):
        if((coords2[i,1] - coords1[i,1])>0 ):        
            R[coord[0],coord[1]] = coords2[i,1] - coords1[i,1]
        else:
            desprezados+=1
        i+=1
            
    print('desprezados =',desprezados)
    return R
def g_map(imd,ime):
    X,Y,Z = np.shape(ime)

    imr,coords1,coords2,coordsave,coordssave2= func(ime,imd)
    coords1 = np.array(coords1)
    coords2 = np.array(coords2)
#    print(np.shape(coords1),np.shape(coords2))
    Dmap = disp_map(ime,coords1,coords2)
#    print(Dmap.max())
    idx = np.where(Dmap>0)
#    for i in range(np.shape(idx)[1]):
#        Dmap[idx[0][i]-2:idx[0][i]+2,idx[1][i]-2:idx[1][i]+2] = Dmap[idx[0][i],idx[1][i]]
#    print(np.max(Dmap))
    seg_im = slic(ime, n_segments=75, compactness=5, sigma=1)
    res = np.zeros((X,Y))
    for i in range(seg_im.max()+1):
        idx = np.where(seg_im==i)
        count = []
        for j in range(np.shape(idx)[1]):
            if(Dmap[idx[0][j],idx[1][j]]>0):
                count.append(Dmap[idx[0][j],idx[1][j]])
        if(np.shape(count)[0]>0):
            median = np.median(count)
#            median = np.mean(count)
#            median = mode(count)
#            median = np.min(count)
#            print(median[0][0])
            res[idx] = median
        else:
    #        print('s')
            res[idx] = -1
#    print(np.max(res))
#    Ot = np.floor(threshold_otsu(res[np.where(res>0)]))
    Ot = threshold_otsu(res[np.where(res>0)])
#    Ot = 21
    print('Otsu: ',Ot)
    trimap = np.zeros((X,Y))
    trimap[np.where(res>Ot)] = trimap[np.where(res>Ot)]+1
    trimap[np.where(res>0)] = trimap[np.where(res>0)]+1
    return Dmap,trimap,res,coordsave,coords1,coords2,coordssave2
def new_meta(im,wid=5):
    
    X,Y,Z = np.shape(im)
    imr = np.zeros((X,Y))
    for i in range(wid,X-wid):
        for j in range(wid,Y-wid):
            imr[i,j] = np.std(im[i-wid:i+wid,j-wid:j+wid])
    
    print('Maximo:',np.max(imr),'Minimo:',np.min(imr),'Media:',np.mean(imr),'Std:',np.std(imr))
    imr = (imr-imr.mean())/(imr.std())
    return imr
def insert_frame(text,T_idx,S_idx,N_coluna, N_linha,N_pixels, vid_idx):
    aux = ''
    aux = aux+str(T_idx)
    aux = aux+','+str(S_idx)
    aux = aux+','+str(N_coluna)
    aux = aux+','+str(N_linha)
    aux = aux+','+str(N_pixels)
    aux = aux+','+str(vid_idx)
    text = text+"insert into imagem values("+aux+ ");\n"
    return text

def insert_pixel(coluna,linha,valor,vid_idx,S_idx,T_idx,coluna_C,linha_C,vid_idx_C,S_idx_C,T_idx_C):
    aux = ''
    aux = aux+str(coluna)
    aux = aux+','+str(linha)
    aux = aux+',array['+str(valor[0])+','+str(valor[1])+','+str(valor[2]) +']'
    aux = aux+','+str(vid_idx)
    aux = aux+','+str(S_idx)
    aux = aux+','+str(T_idx)
    aux = aux+','+str(coluna_C)
    aux = aux+','+str(linha_C)
    aux = aux+','+str(vid_idx_C)
    aux = aux+','+str(S_idx_C)
    aux = aux+','+str(T_idx_C)
    text = ''
    text = text+"insert into pixels values("+aux+");\n"
    return text
##Main
pathD = r'C:\Users\gbz_2\Desktop\UTFPR\TCC\Video1\D'
pathE = r'C:\Users\gbz_2\Desktop\UTFPR\TCC\Video1\E'
pathsave = r'C:\Users\gbz_2\Desktop\UTFPR\TCC\Res'
f = 'frame'
#text = ''
texto = open('texto.txt','w')
texto.flush()
texto.write("insert into video(tempo, nome) values(51, 'video1');\n")

for i in range(51):
    ci = str(i)
#    ime = io.imread(pathE+'\\frame'+ci+'.png')
    
    ime = io.imread('im7.png')
    imd = io.imread('im6.png')
#    imd = io.imread(pathD+'\\frame'+ci+'.png')
    Ax,Ay,Z = np.shape(ime)
    text = ''
    text = insert_frame(text,i,'True',Ax,Ay,Ax*Ay,1)
    texto.write(text)
    text = ''
    text = insert_frame(text,i,'False',Ax,Ay,Ax*Ay,1)
    texto.write(text)
    
    D_map,trimap,res,coordsave,coords1,coords2,coordssave2 = g_map(ime,imd)
    
    X,Y = np.shape(coords1)
#    for j in range(X):
#        text = insert_pixel(coords1[j,0],coords1[j,1],ime[coords1[j][0],coords1[j][1],:],1,'True',i,coords2[j,0],coords2[j,1],1,'False',i)
#        texto.write(text)
#        text = insert_pixel(coords2[j,0],coords2[j,1],imd[coords1[j][0],coords1[j][1],:],1,'False',i,coords1[j,0],coords1[j,1],1,'True',i)
#        texto.write(text)
#    trimap = np.array(127*trimap+1,dtype=np.uint8)
    print('Frame ',i)
#    io.imsave(pathsave+'\\frame'+ci+'.png',trimap)
#    imr = new_meta(ime,wid=5)
    break
#print(text)
texto.close()

#text = np.array(text)
#np.savetxt('Text.out',text)
#plt.figure()
#plt.subplot(211)
#plt.imshow(ime)
#plt.subplot(212)
#plt.imshow(imr,cmap='gray')
#
#plt.figure()
#plt.subplot(211)
#plt.imshow(ime)
#plt.subplot(212)
#plt.imshow(np.where(imr>0.5,1,0),cmap='gray')
#plt.figure()
#plt.subplot(221)
#plt.imshow(ime)
#plt.subplot(222)
#plt.imshow(imd)
#plt.subplot(223)
#plt.imshow(D_map,cmap='gray')
#plt.subplot(224)
#plt.imshow(trimap,cmap='gray')
#
#plt.figure()
#plt.subplot(121)
#plt.hist(np.ndarray.flatten(res[res>0]))
#plt.subplot(122)
#plt.imshow(res,cmap='gray')
#
#plt.figure()
#plt.subplot(121)
#plt.imshow(imd)
#plt.plot([p[1] for p in coordsave],[p[0] for p in coordsave],'.')
#plt.subplot(122)
#plt.imshow(ime)
#plt.plot([p[1] for p in coordssave2],[p[0] for p in coordssave2],'.')
#plt.figure()
#plt.subplot(121)
#plt.imshow(imd)
#plt.plot([p[1] for p in coords1],[p[0] for p in coords1],'.')
#plt.subplot(122)
#plt.imshow(ime)
#plt.plot([p[1] for p in coords2],[p[0] for p in coords2],'.')
#
#plt.figure()
#plt.subplot(121)
#plt.imshow(ime)
#plt.subplot(122)
#plt.imshow(slic(ime, n_segments=75, compactness=5, sigma=1))
#
#plt.figure()
#plt.subplot(121)
#plt.imshow(ime)
#plt.subplot(122)
#plt.imshow(trimap,cmap='gray')
#
#plt.figure()
#plt.subplot(121)
#plt.imshow(trimap,cmap='gray')
#plt.subplot(122)
#plt.imshow(mark_boundaries(imd,np.array(trimap,dtype=np.int64)))
#
#idx = np.where(D_map>0)
##for i in range(np.shape(idx)[1]):
##    D_map[idx[i][0]-1:idx[i][0]+2,idx[i][1]-1:idx[i][1]+2] = D_map[i]
#plt.figure()
#plt.imshow(D_map)
#plt.plot(idx[1],idx[0],'.')
#
#plt.figure()
#plt.imshow(create_harris_matrix(ime),cmap='gray')
##plt.plot([p[1] for p in idx[1]],[p[0] for p in idx[0]],'.')
#
#plt.figure()
#plt.imshow(res,cmap='gray')