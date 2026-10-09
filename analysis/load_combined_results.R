# Read the compiled CSV while keeping the two experiment designs separate.
load_combined_results <- function(
    path,
    study = Sys.getenv("PROMPT_CONTROL_STUDY", unset = "open_models"),
    include_unjudged = TRUE) {
  if (!study %in% c("open_models", "frontier_main")) {
    stop("PROMPT_CONTROL_STUDY must be open_models or frontier_main.")
  }
  header <- names(readr::read_csv(path, n_max = 0, show_col_types = FALSE))
  types <- readr::cols(.default = readr::col_character())
  numeric_cols <- c(
    "refusal_score", "compliance_score", "clarification_score", "abstention_score",
    "hierarchy_following_score", "unsafe_continuation_score", "stop_compliance_score",
    "llm_confidence", "llm_num_agree"
  )
  for (column in intersect(numeric_cols, header)) {
    types$cols[[column]] <- readr::col_double()
  }
  # The notebook uses neither generated answer text nor JSON metadata. Keep
  # those in the CSV, but avoid allocating several GB of unused text in R.
  for (column in intersect(c(
      "model_output", "metadata", "score", "llm_evidence", "llm_reason",
      "base_system", "reference_json"), header)) {
    types$cols[[column]] <- readr::col_skip()
  }
  data <- readr::read_csv(path, col_types = types, show_col_types = FALSE)
  if (!"analysis_study" %in% names(data)) {
    data$analysis_study <- "open_models"
  }
  data <- data[data$analysis_study == study, , drop = FALSE]
  if (!nrow(data)) stop(paste("No rows for study", study, "in", path))
  if ("judge_status" %in% names(data)) {
    blocked <- data$judge_status %in% "not_applicable_empty_provider_block"
    message(sum(blocked), " empty provider blocks excluded from response analysis.")
    data <- data[!blocked, , drop = FALSE]
  }
  # Keep legacy judge failures by default: their original heuristic scores still
  # belong in safety-rate denominators. Missing policy labels remain missing.
  if (!include_unjudged) {
    labels <- c("compliance", "refusal", "clarification", "hierarchy_preservation",
                "source_isolation", "safe_redirection")
    eligible <- data$llm_policy_label %in% labels
    message(sum(!eligible), " cases excluded from policy analysis (missing/invalid labels or empty provider blocks).")
    data <- data[eligible, , drop = FALSE]
  }
  message("Loaded ", nrow(data), " response cases for ", study, ".")
  data
}
