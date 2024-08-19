import React from "react";
import { render, screen, userEvent } from "../../../../../test-utils";
import "@testing-library/jest-dom";
import { FeedbackButtons } from "../FeedbackButtons";
import { SystemChat, Settings } from "../../../../../Types";
import { updateFeedback } from "../../../../../utils/OpenSearch";

const mockSystemChat: SystemChat = new SystemChat({
  interaction_id: "1",
  feedback: null,
  comment: "",
});

const mockSettings: Settings = new Settings({
  activeConversation: "test-conversation",
});

jest.mock("../../../../../utils/OpenSearch", () => ({
  updateFeedback: jest.fn(),
}));

describe("FeedbackButtons", () => {
  it("renders thumbs up and thumbs down icons", () => {
    render(
      <FeedbackButtons systemChat={mockSystemChat} settings={mockSettings} />,
    );
    expect(screen.getByTestId("thumb-up-icon")).toBeInTheDocument();
    expect(screen.getByTestId("thumb-down-icon")).toBeInTheDocument();
  });

  it("renders comment input field", () => {
    render(
      <FeedbackButtons systemChat={mockSystemChat} settings={mockSettings} />,
    );
    expect(screen.getByPlaceholderText("Leave a comment")).toBeInTheDocument();
  });

  it("renders filled thumbs up icon if initial feedback is true", () => {
    const systemChatWithPositiveFeedback = new SystemChat({
      ...mockSystemChat,
      feedback: true,
    });
    render(
      <FeedbackButtons
        systemChat={systemChatWithPositiveFeedback}
        settings={mockSettings}
      />,
    );
    expect(screen.getByTestId("thumb-up-icon-filled")).toBeInTheDocument();
  });

  it("renders filled thumbs down icon if initial feedback is false", () => {
    const systemChatWithNegativeFeedback = new SystemChat({
      ...mockSystemChat,
      feedback: false,
    });
    render(
      <FeedbackButtons
        systemChat={systemChatWithNegativeFeedback}
        settings={mockSettings}
      />,
    );
    expect(screen.getByTestId("thumb-down-icon-filled")).toBeInTheDocument();
  });

  it("updates feedback to thumbs up on click and calls handleSubmit", () => {
    render(
      <FeedbackButtons systemChat={mockSystemChat} settings={mockSettings} />,
    );
    const thumbUpButton = screen.getByTestId("thumb-up-icon");
    userEvent.click(thumbUpButton);
    expect(updateFeedback).toHaveBeenCalledWith(
      "test-conversation",
      "1",
      true,
      "",
    );
  });

  it("updates feedback to thumbs down on click and calls handleSubmit", () => {
    render(
      <FeedbackButtons systemChat={mockSystemChat} settings={mockSettings} />,
    );
    const thumbDownButton = screen.getByTestId("thumb-down-icon");
    userEvent.click(thumbDownButton);
    expect(updateFeedback).toHaveBeenCalledWith(
      "test-conversation",
      "1",
      false,
      "",
    );
  });

  it("resets feedback to null on second click and calls handleSubmit", () => {
    const systemChatWithPositiveFeedback = {
      ...mockSystemChat,
      feedback: true,
    };
    render(
      <FeedbackButtons
        systemChat={systemChatWithPositiveFeedback}
        settings={mockSettings}
      />,
    );
    const thumbUpButton = screen.getByTestId("thumb-up-icon-filled");
    userEvent.click(thumbUpButton);
    expect(updateFeedback).toHaveBeenCalledWith(
      "test-conversation",
      "1",
      null,
      "",
    );
  });

  it("updates comment on input change", () => {
    render(
      <FeedbackButtons systemChat={mockSystemChat} settings={mockSettings} />,
    );
    const commentInput = screen.getByPlaceholderText("Leave a comment");
    userEvent.type(commentInput, "Test comment");
    expect(commentInput).toHaveValue("Test comment");
  });

  it("calls handleSubmit on Enter key press in comment input", () => {
    render(
      <FeedbackButtons systemChat={mockSystemChat} settings={mockSettings} />,
    );
    const commentInput = screen.getByPlaceholderText("Leave a comment");
    userEvent.type(commentInput, "Test comment");
    userEvent.keyboard("{Enter}");
    expect(updateFeedback).toHaveBeenCalledWith(
      "test-conversation",
      "1",
      false,
      "Test comment",
    );
  });
});
