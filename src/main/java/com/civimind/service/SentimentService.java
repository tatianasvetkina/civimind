package com.civimind.service;

import com.civimind.dto.SentimentResponse;
import com.civimind.model.Sentiment;
import com.civimind.model.SentimentModel;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Service
public class SentimentService {

    private final SentimentModel model;

    private static final Pattern TOKEN_PATTERN =
            Pattern.compile("[\\p{L}\\p{N}_]{2,}");

    public SentimentService(SentimentModel model) {
        this.model = model;
    }

    public SentimentResponse analyze(String text) {

        if (text == null || text.isBlank()) {
            throw new IllegalArgumentException(
                    "Текст обращения не должен быть пустым"
            );
        }

        double[] features = vectorize(text);
        double[] scores = new double[model.classes().size()];

        // Логистическая регрессия
        for (int i = 0; i < scores.length; i++) {

            scores[i] = model.intercept()[i];

            for (int j = 0; j < features.length; j++) {
                scores[i] += model.coefficients()[i][j]
                        * features[j];
            }
        }

        double[] probabilities = softmax(scores);

        // Определение наиболее вероятного класса
        int bestIndex = 0;

        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > probabilities[bestIndex]) {
                bestIndex = i;
            }
        }

        Sentiment sentiment = Sentiment.valueOf(
                model.classes().get(bestIndex)
        );

        return new SentimentResponse(
                sentiment,
                probabilities[bestIndex]
        );
    }

    private double[] vectorize(String text) {

        String normalized = text.toLowerCase(Locale.ROOT);

        List<String> tokens = new ArrayList<>();

        Matcher matcher = TOKEN_PATTERN.matcher(normalized);

        while (matcher.find()) {
            tokens.add(matcher.group());
        }

        Map<String, Integer> counts = new HashMap<>();

        // Униграммы
        for (String token : tokens) {
            counts.merge(token, 1, Integer::sum);
        }

        // Биграммы
        for (int i = 0; i < tokens.size() - 1; i++) {

            String bigram = tokens.get(i)
                    + " " + tokens.get(i + 1);

            counts.merge(bigram, 1, Integer::sum);
        }

        double[] vector = new double[model.idf().length];

        // Расчет TF-IDF
        for (Map.Entry<String, Integer> entry : counts.entrySet()) {

            Integer index = model.vocabulary().get(entry.getKey());

            if (index != null) {

                double tf = 1.0 + Math.log(entry.getValue());

                vector[index] = tf * model.idf()[index];
            }
        }

        // L2-нормализация
        double norm = 0.0;

        for (double value : vector) {
            norm += value * value;
        }

        norm = Math.sqrt(norm);

        if (norm > 0) {
            for (int i = 0; i < vector.length; i++) {
                vector[i] /= norm;
            }
        }

        return vector;
    }

    // Преобразование оценок классов в вероятности
    private double[] softmax(double[] scores) {

        double max = Arrays.stream(scores)
                .max()
                .orElse(0.0);

        double[] probabilities = new double[scores.length];

        double sum = 0.0;

        for (int i = 0; i < scores.length; i++) {
            probabilities[i] = Math.exp(scores[i] - max);
            sum += probabilities[i];
        }

        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] /= sum;
        }

        return probabilities;
    }
}