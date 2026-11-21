package com.civimind.dto;

import com.civimind.model.Sentiment;

public record SentimentResponse(
        Sentiment sentiment,
        double confidence
) {
}
