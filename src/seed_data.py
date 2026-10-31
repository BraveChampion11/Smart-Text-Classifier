"""Hand-written seed sentences for the Smart Text Classifier dataset.

Every sentence here was written by hand following the rules in data/LABELS.md.
build_dataset.py expands them with light, class-neutral variations
(greeting/sign-off wrappers, casing, a typo) to make a few hundred rows.

Structure: SEEDS[label] = list of base sentences.
Five service domains (food delivery, bank app, online shop, university
portal, telecom) plus domain-free generic sentences are covered so the
model cannot learn "Complaint == food delivery".

Hard cases are included on purpose:
  * Inquiries without a question mark ("need my account statement")
  * Polite complaints ("I would appreciate it if ... could be looked into")
  * Complaints phrased as questions ("Where is my refund? It has been 2 weeks!")
  * Feedback that contains negative words but demands no fix
  * Feedback phrased as a question ("Why not add dark mode?")
  * "Other" texts that look like inquiries ("What is the meaning of life?")
"""

COMPLAINT = [
    # food delivery
    "My order arrived cold and half the items were missing.",
    "The rider was rude and delivered to the wrong address.",
    "I was charged twice for the same order and nobody is replying.",
    "Delivery took over two hours and the food was spoiled.",
    "I would appreciate it if the delay on my order could be looked into.",
    "wrong food delivered again, this is the third time",
    "Refund for my cancelled order still not received after ten days.",
    "Why has my order not arrived after three hours?",
    # bank / wallet app
    "Money was deducted from my account but the transfer failed.",
    "My card was blocked without any notice and no one can explain why.",
    "I have been waiting a week for my complaint about a wrong deduction to be resolved.",
    "The app keeps crashing every time I try to pay a bill.",
    "Could you please look into the unauthorized charge on my statement, it has been pending for days.",
    "Customer support never picks up my calls.",
    "ATM kept my card and I still have not got a replacement.",
    "Where is my refund? It has been two weeks!",
    # online shop
    "My parcel arrived damaged and the seller is ignoring my messages.",
    "I received the wrong size and the return request was rejected.",
    "The product is nothing like the pictures, completely misleading.",
    "Still no refund even though the return was picked up two weeks ago.",
    "I would be grateful if my missing parcel could be investigated.",
    "item stopped working after two days, want a replacement",
    "Order marked delivered but I never received anything.",
    "Why was I sent a used product?",
    # university portal
    "The portal has been down for two days and I cannot submit my assignment.",
    "My result shows the wrong grade and no one responds to my emails.",
    "I paid my fee but it is still showing as unpaid.",
    "The registration system crashed during course selection and I lost my seat.",
    "Kindly look into the repeated errors on the attendance page, it is affecting my record.",
    "Transcript request pending for a month, this is unacceptable.",
    "login does not work and the helpdesk ignores my tickets",
    # telecom
    "My internet has been very slow for a week despite paying for the full package.",
    "I was charged for a service I never activated.",
    "No signal at my home for three days, please fix this.",
    "Customer care promised a callback two days ago and nobody called.",
    "I would appreciate it if the billing error on my account could be corrected.",
    "SIM blocked without any reason and I cannot make calls.",
    "recharge deducted but balance not added",
    "Why am I being billed for roaming when I never left the city?",
    # generic
    "This is the worst service I have ever experienced.",
    "I have complained three times and nothing has been done.",
    "Nobody responded to my request and I am very disappointed.",
    "Your staff was extremely rude to me today.",
    "I demand a refund immediately, the service was terrible.",
    "The problem is still not fixed after multiple follow-ups.",
    "This is unacceptable, I expect an explanation.",
    "I was promised a callback and nobody called.",
    "Very poor handling of my case, please escalate it.",
    "Waiting for resolution for ten days now, still nothing.",
    "The product stopped working and support is not helping.",
    "I am not happy with how my issue was handled.",
]

INQUIRY = [
    # food delivery
    "What are your delivery charges to Lahore?",
    "How long does delivery usually take?",
    "need to know if you have vegetarian options",
    "Can I change the address after placing an order?",
    "Do you deliver after midnight",
    "Please tell me how to apply a discount code.",
    "Is cash on delivery available in my area?",
    # bank / wallet app
    "What is the daily transfer limit on my account?",
    "How do I update my phone number in the app?",
    "need my account statement for last three months",
    "Which documents are required to open a savings account?",
    "Is there a fee for international transactions?",
    "When will my new debit card be delivered?",
    "Can you explain how the loan interest rate is calculated",
    # online shop
    "What is your return policy?",
    "Does this phone come with a warranty?",
    "need tracking details for my order",
    "How many days does shipping take to Karachi?",
    "Is this item available in black?",
    "Can I pay in installments?",
    "Where can I find the size chart",
    # university portal
    "When is the last date for fee submission?",
    "How can I download my transcript?",
    "need the schedule for midterm exams",
    "Which courses are offered next semester?",
    "Where do I apply for a hostel room?",
    "Can I defer my semester?",
    "What is the procedure to correct my name on the degree",
    # telecom
    "What data packages do you offer?",
    "How do I check my remaining balance?",
    "need the code to activate roaming",
    "Is 5G available in Islamabad?",
    "Can I port my number to your network?",
    "What are the charges for international calls?",
    "How do I upgrade my package",
    # generic
    "What are your working hours?",
    "How can I contact customer support?",
    "need help with resetting my password",
    "Where is your nearest branch located?",
    "Do you offer a student discount?",
    "Can someone tell me how to cancel my subscription?",
    "What is the status of my request?",
    "Is there a mobile app available?",
    "How do I create an account",
    "What payment methods do you accept?",
    "Could you share the pricing details please",
    "Do you provide services on weekends?",
]

FEEDBACK = [
    # food delivery
    "Love the new menu layout, ordering is much faster now.",
    "The food was good but the packaging could be better.",
    "It would be great to have a live map of the rider.",
    "Not bad, but the app has too many pop-up ads.",
    "The delivery rider was very polite and quick.",
    "Please add an option to schedule orders in advance.",
    "Prices are a bit high, though the quality is decent.",
    # bank / wallet app
    "The new fingerprint login is really convenient.",
    "Dark mode would be a great addition to the app.",
    "The interface looks clean but the transaction history is confusing.",
    "I like that notifications arrive instantly after every payment.",
    "The statement download feature is okay, though it feels slow.",
    "Please add support for QR payments.",
    "Overall a decent app, just a bit cluttered.",
    "Why not add a spending summary to the home screen?",
    # online shop
    "Great variety of products and the prices are fair.",
    "Website is easy to browse, but the filters are not very useful.",
    "Packaging was neat and delivery was quick, thank you.",
    "It would help to show customer photos in reviews.",
    "The checkout process is smooth, I really like it.",
    "Product quality is average, nothing special.",
    "Would love a wishlist feature.",
    # university portal
    "The new portal design is much cleaner than the old one.",
    "Lecture recordings are very helpful, thank you.",
    "The mobile view could be improved.",
    "I like the new grade tracker, but the notifications are too frequent.",
    "Please add a calendar that syncs with deadlines.",
    "The library section is useful but the search is slow.",
    "Overall the portal is fine, just needs a faster login.",
    # telecom
    "Call quality has improved a lot recently.",
    "The self-care app is easy to use.",
    "Package prices are slightly high compared to others.",
    "It would be good to get data usage alerts.",
    "Good coverage in the city, but weaker in rural areas.",
    "I like the weekly bundles, they are flexible.",
    "Please offer more night packages.",
    # generic
    "Overall a good experience, keep it up.",
    "The service was satisfactory, nothing more to add.",
    "Your team is very helpful and friendly.",
    "I think the website could use a cleaner layout.",
    "Good quality for the price.",
    "Staff were professional but the waiting area was crowded.",
    "I suggest adding more payment options.",
    "Not perfect, but I appreciate the effort.",
    "Loved the experience, will recommend to friends.",
    "The instructions were clear and easy to follow.",
    "A bit slow at times, otherwise fine.",
    "Would be nice to have a loyalty program.",
]

OTHER = [
    # greetings and closings
    "Good morning",
    "Hello",
    "Hi there",
    "Hey",
    "Good evening everyone",
    "Good night",
    "Bye",
    "See you tomorrow",
    "Have a great day",
    "Hello, is anyone there",
    "How are you?",
    "Hi, this is Ali",
    # thanks / acknowledgements
    "Thanks!",
    "Thank you so much",
    "ok thanks",
    "Okay",
    "lol",
    "Wishing you all the best",
    "Happy Eid Mubarak",
    "Happy new year",
    "Merry Christmas",
    # tests / noise
    "Test message",
    "Just checking if this chat works",
    "asdfgh",
    "Ignore this message",
    "Sorry wrong number",
    "Sent from my iPhone",
    # spam / promotions
    "Win a free iPhone, click here",
    "Congratulations you have won a prize, claim now",
    "Earn money from home, no experience needed",
    "Buy cheap watches online",
    "Click this link to get free followers",
    "Please vote for my page",
    "Selling my old bicycle, message me",
    "Looking for a flatmate in F-7",
    # small talk and off-topic (some look like inquiries)
    "What is the meaning of life?",
    "Who won the cricket match yesterday?",
    "What is the capital of France?",
    "Tell me a joke",
    "How is the weather today?",
    "I am bored",
    "Can you recommend a good movie?",
    "Nice weather today",
    "What is your favorite colour?",
    "My cat is sleeping on my laptop",
    "Anyone up for cricket this weekend?",
    "The match starts at eight",
    "Is it going to rain tomorrow?",
    "Who are you?",
    "I lost my keys somewhere, has anyone seen them?",
]

SEEDS = {
    "Complaint": COMPLAINT,
    "Inquiry": INQUIRY,
    "Feedback": FEEDBACK,
    "Other": OTHER,
}
