# Run from the repo root: Rscript tests/test_analysis_loader.R
source("analysis/load_combined_results.R")
path <- tempfile(fileext = ".csv")
data <- data.frame(
  analysis_study = c("open_models", "open_models", "frontier_main", "frontier_main"),
  llm_policy_label = c("compliance", "parse_error", "refusal", NA_character_),
  judge_status = c("judged", "invalid_judgment", "judged", "not_applicable_empty_provider_block"),
  model_name = c("Gemma", "Gemma", "Gemini", "Gemini"),
  item_id = c("001", "002", "003", "004"),
  compliance_score = c(1, 0, 0, NA_real_),
  refusal_score = c(0, 1, 1, NA_real_),
  malformed_output = NA_character_,
  model_output = "unused large answer text",
  stringsAsFactors = FALSE
)
readr::write_csv(data, path)
open <- load_combined_results(path, "open_models", include_unjudged = FALSE)
frontier <- load_combined_results(path, "frontier_main", include_unjudged = FALSE)
stopifnot(nrow(open) == 1, nrow(frontier) == 1,
          open$item_id == "001", frontier$item_id == "003",
          is.numeric(open$compliance_score), open$compliance_score == 1,
          is.character(open$malformed_output), !"model_output" %in% names(open))
stopifnot(nrow(load_combined_results(path, "open_models")) == 2)
stopifnot(nrow(load_combined_results(path, "frontier_main")) == 1)
stopifnot(inherits(try(load_combined_results(path, "all"), silent = TRUE), "try-error"))
data$analysis_study <- NULL
readr::write_csv(data, path)
stopifnot(nrow(load_combined_results(path)) == 3)
unlink(path)
message("Analysis loader checks passed.")
