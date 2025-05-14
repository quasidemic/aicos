library(tidyverse)
library(ggplot2)

#n_arts_df <- read_csv("C:/Users/kgk/OneDrive - Aalborg Universitet/phd/projekter/aicos/n-art_model-eval.csv")
n_arts_df <- read_csv("C:/Users/kgk/OneDrive - Aalborg Universitet/phd/projekter/aicos/n-art_n-labels_model-eval.csv")

n_arts_df_long <- n_arts_df |> 
  pivot_longer(cols = c(-"metrics", -"accuracy", -"macro avg", -"weighted avg", -"n_articles", -"n_labels" ), 
               names_to = "class", values_to = "score")

avg_mets <- n_arts_df_long |> 
  filter(score > 0,
         metrics != 'support',
         metrics != 'f1-score', 
         class != 'No outcome') |> 
  group_by(metrics, n_articles, n_labels) |> 
  summarize(class_avg = mean(score))

art_class <- avg_mets |> 
  distinct(n_articles, n_labels) |> 
  group_by(n_labels) |> 
  summarize(min_art = min(n_articles))


## Average score over n articles
ggplot(avg_mets, aes(x = n_articles, y = class_avg)) + 
  geom_point(aes(colour = metrics)) + 
  geom_line(aes(colour = metrics)) + 
  theme_minimal() + 
  scale_x_continuous(breaks = seq(5, 100, 5)) + 
  scale_y_continuous(limits = c(0.4,1)) + 
  #geom_vline(xintercept = art_class$min_art, linetype = "dashed", color = "red") + 
  #geom_label(data = art_class, aes(x = min_art, y = 0.45, label = paste(n_labels, 'outcome(s)')), hjust = 0.0) + 
  labs(x = "Number of articles", y = "Average metric score across predicted outcomes", colour = "Metric") + 
  facet_wrap(~as.factor(paste(n_labels, "outcome(s)"))) + 
  ggtitle("Avg. predictive performance based on n articles and n outcomes")


## mets for one label
mets_musco <- n_arts_df_long |> 
  filter(metrics == 'f1-score', 
         class == 'Musculoskeletal and connective tissue outcomes')

## score over n articles for 1 label with increasing other labels
ggplot(mets_musco, aes(x = n_articles, y = score, colour = as.factor(n_labels))) + 
  geom_point() + 
  geom_line() + 
  theme_minimal() + 
  scale_x_continuous(breaks = seq(5, 100, 5)) + 
  scale_y_continuous(limits = c(0.5,1), breaks = seq(0.5, 1, 0.05)) + 
  labs(x = "Number of articles", y = "F1 score", colour = "Total outcomes predicted") + 
  ggtitle("Predictive performance for 'Muscosceletal' outcome")


## Score over n articles per class
n_arts_df_filt <- n_arts_df_long |> 
  filter(!(metrics == 'f1' & score == 0),
         metrics != 'support',
         class != 'No outcome')


ggplot(n_arts_df_filt, aes(x = n_articles, y = score, colour = metrics)) + 
  geom_point() + 
  geom_line() + 
  theme_minimal() + 
  scale_x_continuous(breaks = seq(5, 100, 5)) + 
  scale_y_continuous(limits = c(0.0,1)) + 
  facet_wrap(~class)
  
  

test <- n_arts_df_filt |> 
  filter(class=='Delivery of care')
  