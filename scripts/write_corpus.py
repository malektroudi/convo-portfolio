"""Generate day 8's help-centre corpus.

Kept as a script so the corpus is reproducible and its failure modes are
documented rather than accidental. Run once; the output is committed.

The corpus is synthetic, and the README says so. That is a deliberate trade:
a public help centre would be more impressive to point at but I could not
control what is in it, and the whole point of day 8 is the cases where naive
retrieval goes wrong. Those are built in on purpose:

  NEAR-DUPLICATE   two cancellation policies, standard and private events,
                   with similar wording and different numbers
  SUPERSEDED       a 2026 deposit policy that explicitly replaces an older one
                   still present in another article
  SPLIT            regular hours and holiday hours live in separate articles,
                   so "are you open on Christmas Day" needs the second
  UNANSWERABLE     nothing anywhere mentions EV charging, so a bot that always
                   answers will be caught inventing
"""

from __future__ import annotations

from pathlib import Path

CORPUS = Path(__file__).parent.parent / "src/convo/projects/day08_rag_cites/corpus"

ARTICLES: dict[str, str] = {
    "booking-how-to.md": """# Booking a table

## Online and by phone

You can book a table through the website, through the chat assistant, or by
calling the restaurant directly during opening hours. Bookings open sixty days
ahead for regular service and ninety days ahead for private events.

## What we need from you

We ask for the number of guests, the date, the time, and a name for the
booking. A phone number is optional but helps if we need to reach you about a
delay. We do not ask for card details to hold a regular table.

## Confirmation

You will receive a confirmation by email within a few minutes. If it has not
arrived after an hour, check your spam folder before calling us, as filters
sometimes catch the confirmation address.
""",
    "cancellation-standard.md": """# Cancelling a standard booking

## Notice period

A standard table booking can be cancelled free of charge up to four hours
before the reservation time. Cancel through the link in your confirmation
email, through the chat assistant, or by calling us.

## Late cancellation

Cancelling with less than four hours' notice does not incur a charge for
regular tables, but repeated late cancellations may mean we ask for a card to
hold future bookings.

## Not arriving

If you have not arrived fifteen minutes after your booking time and we have not
heard from you, we may release the table to walk-in guests. See the late
arrival article for how to let us know you are running behind.
""",
    "cancellation-private-events.md": """# Cancelling a private event

## Notice period

A private event booking must be cancelled at least fourteen days before the
event date for a full refund of the deposit. This is different from a standard
table booking, which can be cancelled up to four hours ahead.

## Between fourteen and seven days

Cancelling between fourteen and seven days before the event refunds half of the
deposit. The remaining half covers produce already ordered for your menu.

## Less than seven days

Cancelling with less than seven days' notice means the deposit is not refunded.
We can usually move the booking to another date at no extra cost if you ask
before the seven-day mark.
""",
    "deposit-policy-2026.md": """# Deposit policy (updated March 2026)

## What changed

From 1 March 2026 the private event deposit is a flat 200 CAD regardless of
party size. This replaces the previous per-head deposit of 25 CAD per guest
described in the large groups article, which is no longer in effect.

## When a deposit applies

A deposit applies to private events and to any booking for more than twelve
guests. Standard tables of twelve or fewer do not require a deposit.

## Paying and refunds

The deposit is taken by card when the booking is confirmed and is deducted from
the final bill. Refunds follow the private event cancellation schedule.
""",
    "large-groups.md": """# Large groups

## Up to twelve guests

The dining room seats up to twelve guests at a single table. Bookings of twelve
or fewer can be made through the normal booking flow with no deposit.

## More than twelve

Parties larger than twelve are handled by the events team as a private event.
They can arrange the back room, which seats up to thirty, or a full buyout for
larger numbers.

## Deposits

Historically a deposit of 25 CAD per guest applied to large group bookings.
Please check the current deposit policy article, as this figure was revised in
2026.

## Set menus

Groups over sixteen choose from a set menu rather than ordering individually,
so the kitchen can serve everyone at once.
""",
    "opening-hours.md": """# Opening hours

## Dinner service

We serve dinner from 5pm to 10pm, seven days a week. The kitchen takes its last
order at 9:30pm, and the dining room closes at 11pm.

## Lunch

We do not currently serve lunch. The kitchen opens at 5pm for dinner service
only.

## Bar

The bar opens at 4pm for drinks before dinner and stays open until 11pm. You do
not need a booking to drink at the bar.

## Holidays

Hours change on public holidays and over the winter break. See the holiday
hours article for exact dates.
""",
    "holiday-hours.md": """# Holiday hours

## Closed days

We are closed on Christmas Day, Boxing Day and New Year's Day. The kitchen and
bar are both closed on those dates and bookings cannot be made for them.

## Reduced service

On Christmas Eve and New Year's Eve we serve a single seating at 6pm with a set
menu. Normal a la carte service does not run on those evenings.

## Thanksgiving and Canada Day

We open at the usual 5pm on Thanksgiving and Canada Day, but the dining room
closes an hour earlier than usual at 10pm.

## Return to normal hours

Regular dinner service resumes on 2 January.
""",
    "allergens.md": """# Allergens

## Telling us

Mention any allergy when you book or when you arrive. The kitchen keeps an
allergen matrix for every dish on the menu and the floor team can walk you
through it.

## The fourteen major allergens

We label dishes for the fourteen major allergens, including gluten, dairy,
eggs, fish, shellfish, peanuts, tree nuts, soy, sesame and sulphites.

## Cross-contamination

The kitchen prepares allergen-free orders on separate boards with separate
utensils. We cannot guarantee a completely allergen-free environment, as nuts
and gluten are both used elsewhere in the kitchen.

## Severe allergies

For a severe allergy, please call ahead rather than relying on the chat
assistant, so the kitchen can plan your dishes in advance.
""",
    "menu-vegan.md": """# Vegan and vegetarian menu

## What we offer

A separate vegan menu runs alongside the main menu and changes with the season.
There are usually five vegan mains and three vegan desserts. The vegetarian
options are marked on the main menu rather than kept separately.

## Gluten-free

Most vegan dishes can be made gluten-free. Ask when you order, as the swap
sometimes changes the side rather than the main.

## Telling us ahead

You do not need to tell us in advance to eat vegan, but it helps for groups of
eight or more so the kitchen can prepare enough.

## Allergens

Vegan does not mean allergen-free. Several vegan dishes contain nuts or sesame.
See the allergens article.
""",
    "accessibility.md": """# Accessibility

## Getting in

The dining room is step-free from the street. The main door is 90cm wide and
there is no lip at the threshold.

## Washroom

There is an accessible washroom on the same floor as the dining room, with
grab rails and a 150cm turning circle.

## Seating

We can seat wheelchair users at any table in the main room. The back room is
reached by three steps and is not step-free, so please mention access needs
when booking a private event.

## Assistance animals

Assistance animals are welcome throughout the restaurant, including the bar.
""",
    "parking.md": """# Parking and getting here

## Street parking

There is paid street parking on Dundas, metered until 9pm on weekdays and until
6pm on Saturdays. Sundays are free.

## Garage

A public garage one block east offers evening rates after 5pm. The entrance is
on the side street rather than the main road.

## Transit

The nearest streetcar stop is two minutes' walk and the nearest subway station
is about twelve minutes on foot.

## Bicycles

There is a bike rack directly outside and a second one at the corner.
""",
    "late-arrival.md": """# Running late

## Grace period

We hold your table for fifteen minutes past the booking time. After that we may
release it to walk-in guests if the dining room is full.

## Letting us know

Call or message through the chat assistant if you are running late. If we know
you are coming we will usually hold the table longer, though on a busy Saturday
we cannot promise it.

## Arriving very late

If you arrive more than forty-five minutes after your booking time, the kitchen
may not be able to serve the full menu, as some dishes stop being prepared
towards the end of service.
""",
    "dress-code.md": """# Dress code

## There isn't one

There is no dress code. Guests come in anything from jeans to black tie and all
of it is fine.

## Private events

Hosts of private events sometimes set their own dress code for their guests.
If you are attending someone else's event, ask the host rather than us.
""",
    "payment.md": """# Paying

## Cards and cash

We take all major debit and credit cards, and contactless. We also take cash,
though the floor team may not always have change for large notes late in
service.

## Splitting the bill

We can split a bill up to six ways. For larger groups we ask for a single
payment, as splitting twelve ways at the table slows service for everyone.

## Service charge

A service charge of 18% is added automatically to tables of eight or more. It
is shown separately on the bill and goes to the floor and kitchen teams.

## Gift cards

Gift cards can be used against any bill including the service charge. They do
not expire.
""",
}


def main() -> int:
    CORPUS.mkdir(parents=True, exist_ok=True)
    for name, body in ARTICLES.items():
        # Written with LF explicitly: chunk offsets are byte-for-byte indexes
        # into this text, and a CRLF checkout would shift every one of them.
        (CORPUS / name).write_text(body, encoding="utf-8", newline="\n")
    total = sum(len(b) for b in ARTICLES.values())
    print(f"wrote {len(ARTICLES)} articles, {total:,} characters to {CORPUS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
