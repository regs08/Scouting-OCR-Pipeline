#!/usr/bin/env Rscript

# Function to validate dates
validate_date <- function(date_str) {
  tryCatch({
    # Parse the date string (assuming YYYYMMDD format)
    year <- as.numeric(substr(date_str, 1, 4))
    month <- as.numeric(substr(date_str, 5, 6))
    day <- as.numeric(substr(date_str, 7, 8))
    
    # Create date object
    date <- as.Date(paste(year, month, day, sep = "-"))
    return(list(valid = TRUE, date = date))
  }, error = function(e) {
    return(list(valid = FALSE, date = NULL))
  })
}

# Test dates
test_dates <- c(
  "20242408",  # Specific date (YYYYMMDD)
  "20240824",  # Standard format (YYYYMMDD)
  "20241224",  # December 24, 2024
  "20240229",  # Leap year date
  "20230229",  # Invalid date (not a leap year)
  "20240000",  # Invalid date
  "20240133"   # Invalid date
)

# Create results data frame
results <- data.frame(
  date_str = test_dates,
  valid = logical(length(test_dates)),
  parsed_date = as.Date(character(length(test_dates)))
)

# Test each date
for (i in seq_along(test_dates)) {
  result <- validate_date(test_dates[i])
  results$valid[i] <- result$valid
  if (result$valid) {
    results$parsed_date[i] <- result$date
  }
}

# Print results
cat("Testing date validation:\n")
cat("=======================\n\n")
for (i in seq_along(test_dates)) {
  cat(sprintf("Testing date: %s\n", test_dates[i]))
  if (results$valid[i]) {
    cat(sprintf("  Valid: YES\n"))
    cat(sprintf("  Parsed as: %s\n", format(results$parsed_date[i], "%Y-%m-%d")))
  } else {
    cat("  Valid: NO\n")
    cat("  Error: Invalid date\n")
  }
  cat("\n")
}

# Create visualizations
# 1. Bar plot of validation results
png("date_validation_results.png", width = 800, height = 600)
par(mar = c(5, 4, 4, 2))
barplot(
  height = as.numeric(results$valid),
  names.arg = results$date_str,
  col = ifelse(results$valid, "green", "red"),
  main = "Date Validation Results",
  xlab = "Date Strings",
  ylab = "Validation Status",
  las = 2
)
dev.off()

# 2. Calendar heatmap of valid dates
if (any(results$valid)) {
  library(lubridate)
  valid_dates <- results$parsed_date[results$valid]
  
  # Create a data frame with year, month, and day
  date_df <- data.frame(
    date = valid_dates,
    year = year(valid_dates),
    month = month(valid_dates),
    day = day(valid_dates)
  )
  
  # Create a heatmap
  png("date_heatmap.png", width = 800, height = 600)
  par(mar = c(5, 4, 4, 2))
  plot(
    date_df$month,
    date_df$day,
    pch = 19,
    col = "blue",
    main = "Valid Dates Distribution",
    xlab = "Month",
    ylab = "Day",
    xlim = c(1, 12),
    ylim = c(1, 31)
  )
  grid()
  dev.off()
  
  # 3. Timeline plot
  png("date_timeline.png", width = 800, height = 400)
  par(mar = c(5, 4, 4, 2))
  plot(
    valid_dates,
    rep(1, length(valid_dates)),
    type = "h",
    col = "blue",
    main = "Timeline of Valid Dates",
    xlab = "Date",
    ylab = "",
    yaxt = "n"
  )
  points(valid_dates, rep(1, length(valid_dates)), pch = 19, col = "red")
  dev.off()
} 