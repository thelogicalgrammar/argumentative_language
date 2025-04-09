library(tidyverse)
library(brms)
library(aida)
library(cspplot)

# these options help Stan run faster
options(mc.cores = parallel::detectCores(),
        brms.backend = "cmdstanr")

# global color scheme from CSP
project_colors = cspplot::list_colors() |> pull(hex)

# setting theme colors globally
scale_colour_discrete <- function(...) {
  scale_colour_manual(..., values = project_colors)
}
scale_fill_discrete <- function(...) {
  scale_fill_manual(..., values = project_colors)
}

d_pilot <- read_csv("data_pilot/data_raw.csv") |> 
  mutate(source = "pilot")
d_exp1 <- read_csv("data_experiment1/data.csv") |> 
  mutate(source = "exp1") |> 
  mutate(arraySizeCondition = "wideShort")
d_exp2 <- read_csv("data_experiment2/data.csv") |> 
  mutate(source = "exp2")

# number of participants in the pilot study
d_pilot |> nrow() / 20

d_exps <- bind_rows(d_exp1, d_exp2) |> 
  mutate(source = factor(source, levels = c("exp1", "exp2"))) |> 
  mutate(arrayCondition = case_when(
    arraySizeCondition == "wideShort" ~ " 5 x 12",
    arraySizeCondition == "narrowShort" ~ " 5 x 6",
    arraySizeCondition == "wideLong" ~ "11 x 12",
    arraySizeCondition == "narrowLong" ~ "11 x  6"
  )) |> 
  mutate(condition = ifelse(condition, "high", "low") |> factor(levels = c("high", "low")))

# number of participants in each experiment
d_exps |> count(source) |> mutate(n = n / 20)
d_exps |> count(gender) |> mutate(n = n / 20)
d_exps |> summarize(mean_age= mean(age, na.rm = TRUE),
                    sd_age = sd(age, na.rm = TRUE),
                    min_age = min(age, na.rm = TRUE),
                    max_age = max(age, na.rm = TRUE) )
d_exps |> summarize(
  mean_duration = mean(experiment_duration/60000),
  median_duration = median(experiment_duration/60000)
  )


# number of participants assigned to each size condition
d_exps |> count(arrayCondition) |>  mutate(n = n / 20)

## sum stats & basic plot for aggregate data ----

sum_stats <- d_exps |> 
  group_by(condition, arrayCondition) |> count(responses) |> 
  mutate(proportion = n / sum(n))

sum_stats |> 
  ggplot(aes(x = responses , y = proportion, group = condition, fill = condition)) +
  geom_bar(stat = "identity", position = "dodge") + 
  facet_grid(arrayCondition ~ ., scale = "free") +
  theme_csp() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave("pics/barplot_responses_perConditions.pdf", width = 8, height = 4.5, scale = 1.5)


## sum stats & basic plot for selected situations data ----

sum_stats_full <- d_exps |> 
  group_by(condition, arrayCondition, studentsArray) |> count(responses) |> 
  mutate(proportion = n / sum(n)) 

sum_stats_full |> 
  filter(studentsArray == "12|12|9|3|3") |> 
  ggplot(aes(x = responses , y = proportion, group = condition, fill = condition)) +
  geom_bar(stat = "identity", position = "dodge") + 
  theme_csp() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("pics/barplot_responses_example12|12|9|3|3.pdf", width = 8, height = 4.5, scale = 1)

## regression models ----


# fit_noCondition <- brms::brm(
#   formula = responses ~ arrayAndSize,
#   data = d_exps |> mutate(arrayAndSize = paste(arrayCondition, studentsArray)),
#   family = categorical()
# )
# 
# fit_full <- brms::brm(
#   formula = responses ~ arrayAndSize * condition,
#   data = d_exps |> mutate(arrayAndSize = paste(arrayCondition, studentsArray)),
#   family = categorical()
# )



