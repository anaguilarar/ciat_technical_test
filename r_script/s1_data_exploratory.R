#####
####  Exploratory analysis
#### @Andres Aguilar Ariza
#### 28/04/2024


## library

rm(list= ls())


library(tidyverse)
library(MASS)
library(outliers)


# personal functions for plots
source(
  paste0(
    "https://raw.githubusercontent.com/anaguilarar/R_general_functions/master/scripts//", 
    "plot_functions.R"
  )
)

# functions
transform_usingboxcox <- function(x){
  eps = 1e-5
  lmmodel <- lm(x~1, y = TRUE)
  b <- tryCatch({
    b <- boxcox(lmmodel)
    lambda <- b$x[which.max(b$y)]
    return((x^lambda - 1)/lambda)
  }, error = function(e){
    b <- boxcox(lm(x+eps~1, y = TRUE))
    lambda <- b$x[which.max(b$y)]
    return((x^lambda - 1)/lambda)
        
  })
  
}

## paths

dataset_path = "data/data_base_RF.xlsx"
results_path = "results"

if(!file.exists(results_path)) dir.create(results_path)

# data reading

dataset  = readxl::read_excel(dataset_path)

# rename columns
# remove spaces and symbols
to_remove = c(" ", "/", "[\\(\\)]")

newnames= do.call(c,lapply(names(dataset), function(colname){
  for( i in to_remove){
    colname = unlist(str_replace_all(colname, i, ""))
  }
  return(tolower(colname))
}))

colnames(dataset) = newnames
median_value = dataset%>%
  pull(yieldkgha)%>%
  mean()

std_value = dataset%>%
  pull(yieldkgha)%>%
  sd()

histplot = grafica_rendimiento(dataset%>%
                   data.frame(), y_variable = "yieldkgha", name_y_axes = "Yield (kg/ha)")+
  geom_vline(xintercept = median_value)

ggsave(plot = histplot , filename = file.path(results_path, "yield.png"), width = 10, height = 6, dpi = 400, unit = 'cm')



# per locality and sason

dataset = dataset%>%
  mutate(
    department = unname(sapply(site, function(x)str_split(x , "_", simplify = TRUE)[2])))%>%
  mutate(
    dep_season = paste0(department, "_", season)
  )

grafica_Categorica(dataset%>%
  data.frame(), x_variable = "dep_season", y_variable = "yieldkgha", name_x_axes = "Departament - Season",
  name_y_axes = "Yield (kg/ha)", boxwidth = 0.8)

ggsave(file.path(results_path, "departament_season.png"), width = 12, height = 8, dpi = 400, unit = 'cm')

## average summary
dataset%>%
  group_by(dep_season)%>%
  summarise(meanyield = median(yieldkgha),
            stdyild = sd(yieldkgha))


length(unique(dataset$id))
# per season

grafica_Categorica(dataset%>%
                     data.frame(), x_variable = "season", y_variable = "yieldkgha", name_x_axes = "Season",
                   name_y_axes = "Yield (kg/ha)", boxwidth = 0.8)

ggsave(file.path(results_path, "season.png"), width = 12, height = 8, dpi = 400, unit = 'cm')


## outliers in the input features

input_features = colnames(dataset)[!colnames(dataset) %in%
  c("id", "season","site","yieldkgha","department","dep_season" )]


## finding outliers
feat = "num_hd_tmaxdays" 
outliersdata = do.call(rbind,lapply(input_features, function(feat){
  print(feat)
  data.frame(
    id = dataset$id,
    name = feat,
    value = dataset%>%pull(feat))%>%
    mutate(outlier = ifelse(scores(transform_usingboxcox(x = as.numeric(value)), type= 'chisq', prob = 0.95), 'red', 
                            'blue'))
  
}))


datawithoutliers = dataset%>%
  dplyr::select(id, all_of(input_features))%>%
  pivot_longer(cols = all_of(input_features))%>%
  left_join(outliersdata%>%
              dplyr::select(-value), by =c("id","name") )

## total outliers
datawithoutliers%>%
  filter(outlier == 'red')%>%
  pull(id)%>%
  unique()%>%
  length()


outliersid = datawithoutliers%>%
  filter(outlier == 'red')%>%
  group_by(id)%>%
  summarise(nc = n())%>%
  filter(nc>5)%>%
  pull(id)

print(paste0('total to remove ', length(outliersid)))

outliersplot = datawithoutliers%>%
  ggplot(aes(name, value))+
  geom_boxplot()+
  geom_point(data = datawithoutliers%>%
               filter(id %in% unique(outliersid))%>%
               filter(outlier == 'red'),aes(name, value, color = outlier), alpha = 0.5)+
  facet_wrap(.~ name, scales = "free")+
  theme(
    strip.text.x = element_blank())+
  labs(x = 'Variables')+
  theme_Publication()

ggsave(plot = outliersplot, filename = file.path(results_path, 
                                                 "outliers.png"), width = 24, height = 22, dpi = 400, unit = 'cm')

#
dataset_after = dataset%>%
  filter(! id %in% unique(outliersid))

## correlation plots

corplot = corplotpaper(dataset_after%>%
               dplyr::select(all_of(input_features))%>%
               data.frame(), add_signficance = F, angleaxis = 30, hjust = 0.9, hjustx = 1)

ggsave(plot = corplot, filename = file.path(results_path, 
                                                 "corrplot.png"), width = 24, height = 22, dpi = 400, unit = 'cm')


corplotpaper(dataset_after%>%
               dplyr::select(total_ref_et_all_csmm,total_rice_cwr_all_csmm, yieldkgha)%>%
               data.frame(), add_signficance = F, angleaxis = 30, hjust = 0.9, hjustx = 1)


ggplot(dataset_after, aes(total_ref_et_all_csmm,total_rice_cwr_all_csmm ))+
  geom_point()

ggplot(dataset_after, aes(total_pre_all_csmm,mean_ari_index_all_cs ))+
  geom_point()

features_to_remove = "total_rice_cwr_all_csmm"

## final data set
input_features
dataset_after%>%
  dplyr::select(-all_of(features_to_remove)) %>%
  write.csv(file.path(results_path, "data.csv"), row.names = F)
names(dataset_after)
