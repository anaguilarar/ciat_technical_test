#####
#### Importance variables
#### @Andres Aguilar Ariza
#### 28/04/2024


rm(list= ls())

library(tidyverse)



# personal functions for plots
source(
  paste0(
    "https://raw.githubusercontent.com/anaguilarar/R_general_functions/master/scripts//", 
    "plot_functions.R"
  )
)


## paths

dataset_path = "data/data_base_RF.xlsx"
modeldata_path = "results/data.csv"
mdidata_path = "results/mdi_kfolds10.csv"

results_path = "results"

if(!file.exists(results_path)) dir.create(results_path)


## data reading

datamdi = read.csv(mdidata_path)

orig_data = readxl::read_xlsx(dataset_path)
models_data = read.csv(modeldata_path)

# getting original features names

features = unique(datamdi$features)

orig_names = do.call(rbind,lapply(1:length(names(orig_data)), function(x){
  to_remove = c(" ", "/", "[\\(\\)]")
  colname = names(orig_data)[x]
  for( i in to_remove){
    colname = tolower(unlist(str_replace_all(colname, i, "")))
  }
  nameindex = which(colname == features)
  if(length(nameindex))
    return(data.frame(trname = features[nameindex],
                      orig_name = names(orig_data)[x]))
}))

datamdi = datamdi%>%
  mutate(features_orig = factor(features, levels = orig_names$trname))
levels(datamdi$features_orig) = orig_names$orig_name

##

orderedfeatures = datamdi%>%
  group_by(features_orig)%>%
  summarise(medianmdi = median(mdi))%>%
  arrange(medianmdi)%>%
  pull(features_orig)

eval_metrics = paste0(
  "R² = ", round(mean(datamdi$r2), 2), "\n",
  "RMSE = ", round(mean(datamdi$rmse), 2)
)
datamdi %>%
  mutate(features_orig = factor(features_orig, levels=  orderedfeatures))%>%
  ggplot(aes(x = mdi, y = features_orig), fill =NA)+
  geom_boxplot()+
  geom_point(alpha = 0.3)+
  labs(subtitle = eval_metrics, x = "Mean decrease in impurity (MDI)",
       y = "Variables")+
  theme_Publication()

ggsave(file.path(results_path, 
                                            "mdi.png"), 
       width = 16, height = 15, dpi = 400, unit = 'cm')


## 

mostimportantvariables = orderedfeatures[(length(orderedfeatures)-1):length(orderedfeatures)]

orig_data%>%
  dplyr::select(ID, all_of(mostimportantvariables), `Yield (kg/ha)`)%>%
  pivot_longer(cols = all_of(mostimportantvariables))%>%
  ggplot()+
  geom_point(aes(value, `Yield (kg/ha)`))+
  facet_wrap(.~name, scales = 'free_x')+
  theme_Publication()
  
ggsave(file.path(results_path, 
                 "most_important_variables.png"), 
       width = 16, height = 8, dpi = 400, unit = 'cm')


