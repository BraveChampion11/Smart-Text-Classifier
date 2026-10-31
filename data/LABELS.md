# Label definitions

| Label | Definition |
|---|---|
| **Complaint** | The writer reports a problem or bad experience and wants it fixed or acknowledged. |
| **Inquiry** | The writer asks for information or help (price, hours, status, how-to). |
| **Feedback** | The writer gives an opinion or suggestion about a product or service, without demanding a fix. |
| **Other** | Greetings, thanks, spam, small talk, or anything unrelated to a service or product. |

## Tie-breaker rules (labelling by dominant intent)

1. Complaint beats Inquiry when both are present: "Where is my order? It's been 2 weeks!" -> Complaint.
2. Inquiry beats Feedback when the person asks something: "Love the app, can I export my data?" -> Inquiry.
3. A negative opinion with no demand for a fix -> Feedback ("The app is a bit slow but fine").
4. A plain "thank you" or "hello" with no content -> Other.
