package com.civimind.controller;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@SpringBootTest
@AutoConfigureMockMvc
class SentimentControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Test
    void shouldAnalyzeGetRequest() throws Exception {

        mockMvc.perform(get("/api/sentiment")
                        .param("text", "Спасибо за прекрасную работу"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.sentiment").value("POSITIVE"))
                .andExpect(jsonPath("$.confidence").isNumber());
    }

    @Test
    void shouldAnalyzePostRequest() throws Exception {

        mockMvc.perform(post("/api/sentiment")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {
                                  "text": "Возмущена качеством обслуживания"
                                }
                                """))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.sentiment").value("NEGATIVE"))
                .andExpect(jsonPath("$.confidence").isNumber());
    }

    @Test
    void shouldRejectEmptyText() throws Exception {

        mockMvc.perform(get("/api/sentiment")
                        .param("text", ""))
                .andExpect(status().isBadRequest());
    }

    @Test
    void shouldRejectMissingText() throws Exception {

        mockMvc.perform(get("/api/sentiment"))
                .andExpect(status().isBadRequest());
    }

    @Test
    void shouldRejectLongText() throws Exception {

        mockMvc.perform(post("/api/sentiment")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"text\":\""
                                + "а".repeat(10001) + "\"}"))
                .andExpect(status().isBadRequest());
    }

    @Test
    void shouldRejectInvalidJson() throws Exception {

        mockMvc.perform(post("/api/sentiment")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{invalid}"))
                .andExpect(status().isBadRequest());
    }
}
