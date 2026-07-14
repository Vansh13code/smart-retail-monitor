from app.services.product_services import ProductService

detections = [

    {
        "id":1,
        "class":"Shelf",
        "bbox":[0,0,500,250]
    },

    {
        "id":2,
        "class":"milk",
        "bbox":[100,100,150,180]
    },

    {
        "id":3,
        "class":"snacks",
        "bbox":[220,100,280,180]
    },

    {
        "id":4,
        "class":"Shelf",
        "bbox":[0,260,500,500]
    },

    {
        "id":5,
        "class":"Shampoo",
        "bbox":[100,300,150,420]
    }

]

service = ProductService()

inventory = service.process(detections)

print(inventory)