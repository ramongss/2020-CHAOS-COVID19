rm(list = ls())
setwd("C:/Users/Usuario/Dropbox/COVID-19")

#Save several directories
BaseDir       <- getwd()
CodesDir      <- paste(BaseDir, "Codes", sep="/")
FiguresDir    <- paste(BaseDir, "Figures", sep="/")
ResultsDir    <- paste(BaseDir, "Results", sep="/")
DataDir       <- paste(BaseDir, "Data",sep="/")

#Load Packages
setwd(CodesDir)
source("checkpackages.R")
source("elm_caret.R")
source("Metricas.R")


packages<-c("forecast","cowplot","Metrics","caret","elmNNRcpp","tcltk","TTR",
            "foreach","iterators","doParallel","lmtest","wmtsa","httr",
            "jsonlite","magrittr")

sapply(packages,packs)

library(extrafont)
windowsFonts(Times = windowsFont("TT Times New Roman"))
library(ggplot2)
library(Cairo)

#Checa quantos núcleos existem
ncl<-detectCores();ncl

#Registra os clusters a serem utilizados
cl <- makeCluster(ncl-1);registerDoParallel(cl)

#########################################################################
# Load the data from API
covid <- 
  GET(url = "https://brasil.io/api/dataset/covid19/caso/data/?place_type=state") %>% 
  content %>%
  toJSON %>%
  fromJSON

# Separate the data
covid <- data.frame(
  date        = covid$results$date      %>% unlist %>% as.Date,
  state       = covid$results$state     %>% unlist,
  confirmed   = covid$results$confirmed %>% unlist,
  deaths      = covid$results$deaths    %>% unlist
)

# Order by state and date
covid <- covid[order(covid$state, covid$date),]; rownames(covid) <- NULL

# Plot confirmed cases by state
covid %>% 
  ggplot(aes(x = date)) +
  geom_line(aes(y = confirmed)) +
  facet_wrap(~state, scales = "free")+
  scale_x_date(date_labels = "%d/%m")


# States by list
covid_list <- covid %>%
  split(covid$state)

covid_list_states<-covid_list[States]
#Insert incidence
for(i in 1:length(covid_list))
{
  covid_list[[i]]$incidence <-rep(0,times=dim(covid_list[[i]])[1])
  covid_list[[i]]$dincidence<-rep(0,times=dim(covid_list[[i]])[1])
  covid_list[[i]]$rincidence <-rep(0,times=dim(covid_list[[i]])[1])
  covid_list[[i]]$rdincidence<-rep(0,times=dim(covid_list[[i]])[1])
  
  for(j in 1:dim(covid_list[[i]])[1])
  {
    if(j==1)
    {
      covid_list[[i]][j,"incidence"] <-covid_list[[i]][j,"confirmed"]
      covid_list[[i]][j,"dincidence"]<-covid_list[[i]][j,"deaths"]
      covid_list[[i]][j,"rincidence"]<-covid_list[[i]][j,"incidence"]/covid_list[[i]][j,"confirmed"]
      covid_list[[i]][j,"rdincidence"]<-covid_list[[i]][j,"dincidence"]/covid_list[[i]][j,"deaths"]
   }
    else
    {
      covid_list[[i]][j,"incidence"] <-covid_list[[i]][j,"confirmed"]-covid_list[[i]][j-1,"confirmed"]
      covid_list[[i]][j,"dincidence"]<-covid_list[[i]][j,"deaths"]-covid_list[[i]][j-1,"deaths"]
      covid_list[[i]][j,"rincidence"]<-covid_list[[i]][j,"incidence"]/covid_list[[i]][j,"confirmed"]
      covid_list[[i]][j,"rdincidence"]<-covid_list[[i]][j,"dincidence"]/covid_list[[i]][j,"deaths"]
    }
  }
}

#Incidence and death incidence plots
covid2<-do.call(rbind.data.frame,covid_list)
# Plot confirmed cases by state
covid2 %>% 
  ggplot(aes(x = date)) +
  geom_line(aes(y = incidence)) +
  facet_wrap(~state, scales = "free")+
  scale_x_date(date_labels = "%d/%m")

covid2 %>% 
  ggplot(aes(x = date)) +
  geom_line(aes(y = dincidence)) +
  facet_wrap(~state, scales = "free")+
  scale_x_date(date_labels = "%d/%m")

#save(covid_list, file = 'covid_state.RData')
#write.csv(covid, 'covid_state.csv', row.names = FALSE)

#Data for analysis
States<-sort(c("PR","SC","RS","SP","RJ","MG","BA","RN","CE","AM"))

data<-covid_list[States]
data_plot<-do.call(rbind.data.frame,data)

# write.xlsx(data_plot,file="Covid_2004.xlsx")
data_plot %>% 
  ggplot(aes(x = date)) +
  geom_line(aes(y = confirmed)) +
  facet_wrap(~state, scales = "free",nrow=2)+
  scale_x_date(date_labels = "%d/%m")

###########################Objects for Modeling#############
setwd(DataDir)
data<-read.table(file="Covid_2004.csv",header=TRUE,sep=";",dec=",")
data<-data[,-1]
data<-split(data,data$state)
data<-data[States]

#Convertind Excel date for R date

for(i in 1:10)
{
  data[[i]][,'date']<-as.Date(data[[i]][,'date'], format = "%m/%d/%Y")
}
#Objetos para treinamento
data_m  <-list();  #Objeto para receber os dados para cada treinamento
Models  <-list();  #Objeto para receber os modelos treinados
Arimas  <-list();
Params  <-list();  #Recebe os parâmetros
models  <-c("cubist", "svmLinear","ridge","rf") #Modelos a serem usados
k               <-1        #Contador
a               <-1
Aux_1           <-matrix(rep(0,times=25),nrow=5,ncol=5,byrow=TRUE)
Aux_2           <-rep(0,times=2)
Ptrain          <-list();               Ptest<-list();
Etrain          <-list();               Etest<-list();
Metrics         <-matrix(nrow=6,ncol=4)
colnames(Metrics)<-c("RRMSE","MAE","SMAPE","SD")
row.names(Metrics)<-c("CUBIST","SVR","RIDGE","RF","Stack","ARIMA")
Metrics_States<-list()
horizontes<-c(1,3,6)
#########################Modeling#####################
cat("\014")

for(H in 1:length(horizontes)) #Variando o Horizonte de previsão
{
  Horizon<-horizontes[H]
  
  for(s in 1:length(data))           #Variando o conjunto de dados
  {
    {
      Data_state      <-c(Aux_2,data[[s]][,"confirmed"])
      
      #--------------------------Construindo Conjuno-----------------------------#
      data_m<-lags(Data_state, n = 5) #n representa o número de lags
      
      colnames(data_m)<-c("Confirmed",paste("Lag",1:5,sep=""))
      
      #----------------------Divisão em treinamento e teste---------------------#
      n      <-dim(data_m)[1]    #Número de observações
      cut    <-n-6               #Ponto de corte entre treino e teste
      
      
      ptrain          <-matrix(nrow=cut,ncol=length(models)+2); #Recebe predições 3SA das componentes
      ptest           <-matrix(nrow=n-cut,ncol=length(models)+2);  #Recebe predições 3SA das componentes
      colnames(ptrain)<-c(models,"GP-Stack","ARIMA");colnames(ptest) <-c(models,"ELM-Stack","ARIMA")
      
      errorstrain          <-matrix(nrow=cut,ncol=length(models)+2); #Recebe predições 3SA das componentes
      errorstest          <-matrix(nrow=n-cut,ncol=length(models)+2);  #Recebe predições 3SA das componentes
      colnames(errorstrain )<-c(models,"GP-Stack","ARIMA");colnames(errorstest) <-c(models,"ELM-Stack","ARIMA")
      
      fitControl2<- trainControl(method= "cv",number=5,savePredictions="final") 
      
      #Train and Test
      train  <-data_m[1:cut,];
      Y_train<-train[,1];X_train<-train[,-1]
      test   <-tail(data_m,n-cut)
      Y_test <-test[,1]; X_test <-test[,-1]
      
      #----------------------Divisões para Treinamento----------------------------#
      
      
      
    }
    
    #-----------------------Training---------------------------#
    for(i in 1:(length(models)+2)) #Aqui está indo de 1 até número de modelos +1 pois tem o stacking
    {
      options(warn=-1)
      if(i != 6)
      {
        if(i != 5)
        {
          set.seed(1234)
          Models[[k]]<-train(as.data.frame(X_train[1:cut,]),as.vector(Y_train),
                             method=models[[i]],
                             preProcess = c("center","scale"), #Processamento Centro e Escala
                             tuneLength= 4,                    #Número de tipos de parâmetros 
                             trControl = fitControl2,verbose=FALSE)
          
        }
        else 
        {
          
          #Stacking ensemble
          modelst <- caretList(as.data.frame(X_train[1:cut,]),as.vector(Y_train),
                               trControl=fitControl2, 
                               preProcess = c("center","scale"),
                               methodList=models)
          Models[[k]] <-caretStack(modelst, 
                                   trControl=fitControl2, 
                                   method='gaussprLinear',
                                   preProcess = c("center","scale"),
                                   tuneLength= 5)
          
        }
        
        #-----------------------Salvando Parâmetros--------------------------------#
        Params[[k]]<-Models[[k]]$bestTune
        
      }
      else
      {
        Arimas[[a]]<-auto.arima(Y_train)
      }
      
      
      #---------------------------Lags-Names-----------------------------------------#
      Lag1<-match("Lag1",colnames(X_test));Lag2<-match("Lag2",colnames(X_test))
      Lag3<-match("Lag3",colnames(X_test));Lag4<-match("Lag2",colnames(X_test))
      Lag5<-match("Lag3",colnames(X_test))
      #------------------Recursive Forecasting for train and test sets-----------
      #Aqui, o conjunto de análise é dividido em n conjuntos de h observações. Nesse caso
      #A cada 3 predições, é reiniciado o processo de previsão. Caso isso não seja feito,
      #as predições continuarão a ser atualizadas e o erro carregado.
      #Se desejar h>3, basta tirar fazer H<-HORIZONTE DESEJADO e descomentar  
      #X_trainm[p+3,Lag3]<-ptrain[p,m] e #X_testm[p+3,Lag3]<-ptest[p,m]
      
      if(i != 6)
      {
        #Vai colocar as predições em cada coluna
        if(Horizon==1)
        {
          h<-Horizon
          
          #Train
          ptrain[1:cut,i]<-round(predict(Models[[k]], X_train[1:cut,]))
          #Test 
          ptest[1:6,i]<-round(predict(Models[[k]],X_test[1:6,]))
          
        }
        else if(Horizon==3)
        {
          h<-Horizon
          #Treinando e prevendo cada componente com cada modelo
          
          X_trainm<-rbind(train[,-1],Aux_1);
          X_testm <-rbind(test[,-1],Aux_1);
          colnames(X_trainm)=colnames(X_train)
          colnames(X_testm) =colnames(X_test)
          #Train
          for(p in 1:cut)
          {
            if(p%%h !=1) #Sempre reinicia na divisão de resto 1-->Multiplos de h+1
            {
              ptrain[p,i]<-round(predict(Models[[k]], as.data.frame(t(X_trainm[p,]))))
              X_trainm[p+1,Lag1]<-ptrain[p,i]
              X_trainm[p+2,Lag2]<-ptrain[p,i]
              
            }
            else
            {
              X_trainm[p:(n-cut),]<-X_train[p:(n-cut),]
              ptrain[p,i]       <-round(predict(Models[[k]], as.data.frame(t(X_trainm[p,]))))
              X_trainm[p+1,Lag1]<-ptrain[p,i]
              X_trainm[p+2,Lag2]<-ptrain[p,i]
              
            }
          }
          #Test  
          for(p in 1:(n-cut))
          {
            if(p%%h !=1)
            {
              ptest[p,i]<-round(predict(Models[[k]], as.data.frame(t(X_testm[p,]))))
              X_testm[p+1,Lag1]<-ptest[p,i]
              X_testm[p+2,Lag2]<-ptest[p,i]
              
              
            }
            else
            {
              X_testm[p:(n-cut),]<-X_test[p:(n-cut),]
              ptest[p,i]       <-round(predict(Models[[k]], as.data.frame(t(X_testm[p,]))))
              X_testm[p+1,Lag1]<-ptest[p,i]
              X_testm[p+2,Lag2]<-ptest[p,i]
              
            }
          }
        }
        else
        {
          #Treinando e prevendo cada componente com cada modelo
          
          X_trainm<-rbind(train[,-1],Aux_1);
          X_testm <-rbind(test[,-1],Aux_1);
          colnames(X_trainm)=colnames(X_train)
          colnames(X_testm) =colnames(X_test)
          h<-Horizon
          
          #Train
          for(p in 1:cut)
          {
            if(p%%h !=1) #Sempre reinicia na divisão de resto 1-->Multiplos de h+1
            {
              ptrain[p,i]<-round(predict(Models[[k]], as.data.frame(t(X_trainm[p,]))))
              X_trainm[p+1,Lag1]<-ptrain[p,i]
              X_trainm[p+2,Lag2]<-ptrain[p,i]
              X_trainm[p+3,Lag3]<-ptrain[p,i]
              X_trainm[p+4,Lag4]<-ptrain[p,i]
              X_trainm[p+5,Lag5]<-ptrain[p,i]
              
            }
            else
            {
              X_trainm[p:(n-cut),]<-X_train[p:(n-cut),]
              ptrain[p,i]       <-round(predict(Models[[k]], as.data.frame(t(X_trainm[p,]))))
              X_trainm[p+1,Lag1]<-ptrain[p,i]
              X_trainm[p+2,Lag2]<-ptrain[p,i]
              X_trainm[p+3,Lag3]<-ptrain[p,i]
              X_trainm[p+4,Lag4]<-ptrain[p,i]
              X_trainm[p+5,Lag5]<-ptrain[p,i]
              
            }
          }
          #Test  
          for(p in 1:(n-cut))
          {
            if(p%%h !=1)
            {
              ptest[p,i]<-round(predict(Models[[k]], as.data.frame(t(X_testm[p,]))))
              X_testm[p+1,Lag1]<-ptest[p,i]
              X_testm[p+2,Lag2]<-ptest[p,i]
              X_testm[p+3,Lag3]<-ptest[p,i]
              X_testm[p+4,Lag4]<-ptest[p,i]
              X_testm[p+5,Lag5]<-ptest[p,i]
              
            }
            else
            {
              X_testm[p:(n-cut),]<-X_test[p:(n-cut),]
              ptest[p,i]       <-round(predict(Models[[k]], as.data.frame(t(X_testm[p,]))))
              X_testm[p+1,Lag1]<-ptest[p,i]
              X_testm[p+2,Lag2]<-ptest[p,i]
              X_testm[p+3,Lag3]<-ptest[p,i]
              X_testm[p+4,Lag4]<-ptest[p,i]
              X_testm[p+5,Lag5]<-ptest[p,i]
            }
          }
        }
        
      }
      else
      {
        predictions_arima<-forecast(Arimas[[a]],h=6)
        ptest[,i] <-round(c(predictions_arima$mean))
        ptrain[,i] <-round(c(Arimas[[a]]$fitted))
        
        }
      
      #Erros
      errorstest[,i] <-round(Y_test-ptest[1:6,i])
      errorstrain[,i] <-round(Y_train-ptrain[1:cut,i])
      
      
      #Metrics in test
      criterias<-PM(Y_test,ptest[1:6,i],mean(Y_test))
      Metrics[i,]<-c(criterias[,c("RRMSE","MAE","SMAPE")],sd(errorstest[1:6,i]))
      
      #Cada elemento da lista recebera uma combinação de process com control
      
      cat("State:",States[s],"Horizon:",Horizon,
          sprintf('RRMSE: %0.3f',Metrics[i,1]),
          sprintf('SMAPE: %0.3f',Metrics[i,3]),
          sprintf('MAE: %0.3f'  ,Metrics[i,2]),
          sprintf('SD: %0.3f'   ,Metrics[i,4]),
          'Model:',ifelse(i==6,"Arima",ifelse(i==5,"Stack-ELM",models[i])),
          "\n")
      
      k<-k+1
    }
    a<-a+1
    
    Metrics_States[[s]]  <-Metrics[order(Metrics[,3],decreasing=FALSE),]
    Ptrain[[s]]          <-ptrain;         
    Ptest[[s]]           <-ptest;
    Etrain[[s]]          <-errorstrain;
    Etest[[s]]           <-errorstest;
  }
  
  names(Metrics_States)<-States
  
  Results<-list(Ptrain,Ptest,Etest,Etrain,Params,Models,Arimas,Metrics_States)
 
  name   <-paste("Results_",Horizon,"SA_",Sys.Date(),".RData", sep='')
 
  setwd(ResultsDir)
  save(Results,file=name)
  #save performance out-of-sample
  name_m<-paste("Metrics_",Horizon,"SA_",Sys.Date(),".xlsx", sep='')

  Result<-do.call(rbind.data.frame,Metrics_States)
  setwd(ResultsDir)
  write.xlsx(Result,file=name_m)
  
  }

#########################Results###########################
setwd(ResultsDir)

load("Results_1SA_2020-04-22.RData");C1SA<-Results
load("Results_3SA_2020-04-22.RData");C3SA<-Results
load("Results_6SA_2020-04-22.RData");C6SA<-Results

models_names<-c("CUBIST","SVR","RIDGE","RF","Stacking","ARIMA")

for(i in 1:10)
{
  colnames(C1SA[[1]][[i]])<-models_names;  colnames(C1SA[[2]][[i]])<-models_names
  colnames(C3SA[[1]][[i]])<-models_names;  colnames(C3SA[[2]][[i]])<-models_names
  colnames(C6SA[[1]][[i]])<-models_names;  colnames(C6SA[[2]][[i]])<-models_names
  
  colnames(C1SA[[4]][[i]])<-models_names;  colnames(C1SA[[3]][[i]])<-models_names
  colnames(C3SA[[4]][[i]])<-models_names;  colnames(C3SA[[3]][[i]])<-models_names
  colnames(C6SA[[4]][[i]])<-models_names;  colnames(C6SA[[3]][[i]])<-models_names
  
}

#Best Models Names
{
  Names1SA<-c(sapply(C1SA[[8]],
                     function(x){rownames(x)[which.min(apply(x,MARGIN=1,min))]}))
  Names1SA<-c("CUBIST","ARIMA","ARIMA","SVR","Stacking","Stacking","ARIMA","SVR","Stacking","SVR")
  Names2SA<-c(sapply(C3SA[[8]],
                     function(x){rownames(x)[which.min(apply(x,MARGIN=1,min))]}))
  Names2SA<-c("CUBIST","SVR","ARIMA","SVR",  "Stacking" ,"Stacking", "ARIMA",    "SVR",  "ARIMA",    "SVR")
  
  Names3SA<-c(sapply(C6SA[[8]],
                     function(x){rownames(x)[which.min(apply(x,MARGIN=1,min))]}))
  Names3SA<-c("RIDGE",    "SVR",  "RIDGE",    "SVR",  "Stacking", 
              "Stacking", "CUBIST",    "SVR",  "Stacking",    "SVR" )
  
}

table(c(Names1SA,Names2SA,Names3SA))

#Models for each state
Best_Models<-list()
Conf_Interv<-list()
for(i in 1:10)
{
  aux<-matrix(nrow=dim(C1SA[[1]][[i]])[1]+6,ncol=6)
  SD<-sd(C1SA[[4]][[i]][,Names1SA[i]])
  
  for(j in 1:3)
  {
    if(j==1)
    {
      aux[,j]<-c(C1SA[[1]][[i]][,Names1SA[i]],C1SA[[2]][[i]][,Names1SA[i]])
      #aux[,j]<-ifelse(v1<0,0,v1)
      
    }
    else if(j==2)
    {
      aux[,j]<-c(C3SA[[1]][[i]][,Names2SA[i]],C3SA[[2]][[i]][,Names2SA[i]])
      #aux[,j]<-ifelse(v1<0,0,v1)
    }
    else
    {
      aux[,j]<-c(C6SA[[1]][[i]][,Names3SA[i]],C6SA[[2]][[i]][,Names3SA[i]])
      #aux[,j]<-ifelse(v1<0,0,v1)
    }
  }
  aux[,4]<-data[[i]][4:dim(data[[i]])[1],"confirmed"]
  LB<-aux[,1]-1.96*SD  #Lower boundarie
  UB<-aux[,1]+1.96*SD  #Lower boundarie
  
  aux[,5]<-ifelse(LB<0,0,round(LB))
  aux[,6]<-ifelse(UB<0,0,round(UB))  #Lower boundarie
  
  aux1<-data.frame(Confirmed=c(aux[,1:4]),
                   Models=rep(c(outer(c("ODA-"),Names1SA[i], FUN = "paste0"),
                                outer(c("TDA-"),Names2SA[i],  FUN = "paste0"),
                                outer(c("SDA-"),Names3SA[i], FUN = "paste0"),
                                "Observed"),each=dim(aux)[1]),
                                Date=rep(data[[i]][4:dim(data[[i]])[1],"date"],times=4),
                   LB=rep(ifelse(LB<0,0,round(LB)),times=4),
                   UB=rep(ifelse(UB<0,0,round(UB)),times=4))
  Conf_Interv[[i]]<-tail(aux,6)
  Best_Models[[i]]<-aux1
}
names(Best_Models)<-States
names(Conf_Interv)<-States

Conf_Interv<-do.call(rbind.data.frame,Conf_Interv)
colnames(Conf_Interv)<-c("OSA","TSA","SDA","Observed","LB","UB")
#Predicted versus observed
setwd(CodesDir)
source("Covid_Plot.R")
setwd(FiguresDir)

for(i in 1:10)
{
  nameps<-paste("PO_",States[i],".eps",sep="")
  
  PO_Covid(Best_Models[[i]])
  
  ggsave(nameps, device=cairo_ps,width = 11,height = 9,dpi = 1200)
  
}

names(C1SA[[3]])<-States;names(C3SA[[3]])<-States;names(C6SA[[3]])<-States
#Grouping Errors
Errors_Models<-data.frame(Error=unlist(C3SA[[3]]),
                          State=rep(States,each=36),
                          Model=rep(models_names,each=6))

ggplot(Errors_Models, aes(x=Model, y=abs(Error))) + 
  #geom_violin(trim=FALSE)+
  #scale_y_log10()+
  geom_boxplot(width = 0.9)+  
  xlab("Model") +  ylab("Absolute error")+
  theme_bw(base_size = 20,base_family="Times New Roman")+
  theme(legend.direction = "none",
        #axis.text=element_text(size=10),
        axis.text.y = element_text(angle = 90,size=12),
        axis.text.x = element_text(angle = 90,size=15),
        #text=element_text(family="Times New Roman"),
        axis.title=element_text(size=20))+
  stat_summary(fun=mean, geom="point", size=2, color="black")+
  facet_wrap(~State, scales = "free",nrow=2)

setwd(FiguresDir)
ggsave("Error_Boxplot3h.eps", device=cairo_ps,width = 11,height = 9,dpi = 1200)
