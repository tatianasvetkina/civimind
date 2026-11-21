package com.civimind.service;

import com.civimind.dto.SentimentResponse;
import com.civimind.model.Sentiment;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
class SentimentServiceTest {

    @Autowired
    private SentimentService sentimentService;

    @Test
    void shouldDetectPositiveSentiment() {

        SentimentResponse response = sentimentService.analyze(
                "Спасибо за прекрасную работу"
        );

        assertEquals(Sentiment.POSITIVE, response.sentiment());
        assertEquals(0.7812, response.confidence(), 0.0001);
    }

    @Test
    void shouldDetectNeutralSentiment() {

        SentimentResponse response = sentimentService.analyze(
                "Прошу сообщить время приема"
        );

        assertEquals(Sentiment.NEUTRAL, response.sentiment());
        assertEquals(0.8839, response.confidence(), 0.0001);
    }

    @Test
    void shouldDetectNegativeSentiment() {

        SentimentResponse response = sentimentService.analyze(
                "Возмущена качеством обслуживания"
        );

        assertEquals(Sentiment.NEGATIVE, response.sentiment());
        assertEquals(0.7279, response.confidence(), 0.0001);
    }

    @Test
    void shouldRejectEmptyText() {

        assertThrows(
                IllegalArgumentException.class,
                () -> sentimentService.analyze("")
        );
    }

    @Test
    void shouldRejectNullText() {

        assertThrows(
                IllegalArgumentException.class,
                () -> sentimentService.analyze(null)
        );
    }

    @Test
    void shouldReturnValidConfidence() {

        SentimentResponse response = sentimentService.analyze(
                "Обращение направлено на рассмотрение"
        );

        assertTrue(response.confidence() >= 0.0);
        assertTrue(response.confidence() <= 1.0);
    }
}