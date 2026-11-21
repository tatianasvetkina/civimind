package com.civimind.model;

import java.util.List;
import java.util.Map;

public record SentimentModel(
        String name,
        String version,
        String algorithm,
        int trainingSamples,
        List<String> classes,
        Map<String, Integer> vocabulary,
        double[] idf,
        double[][] coefficients,
        double[] intercept,
        boolean lowercase,
        int[] ngramRange,
        boolean sublinearTf,
        String tokenPattern,
        String norm
) {
}